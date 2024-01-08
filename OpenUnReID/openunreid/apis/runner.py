# Obtained from: https://github.com/open-mmlab/OpenUnReID

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor


import collections
import os.path as osp
import time
import warnings
import numpy as np
import torch, torchvision
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import GradScaler, autocast
    amp_support = True
except:
    amp_support = False

from ..core.label_generators import LabelGenerator
from ..core.metrics.accuracy import accuracy
from ..data import build_train_dataloader, build_val_dataloader, build_test_dataloader
from ..data.utils.data_utils import save_image
from ..utils import bcolors
from ..utils.dist_utils import get_dist_info, synchronize
from ..utils.meters import Meters
from ..utils.image_pool import ImagePool
from ..utils.torch_utils import copy_state_dict, load_checkpoint, save_checkpoint
from ..utils.file_utils import mkdir_if_missing
from ..utils.torch_utils import tensor2im
from .test import val_reid, test_reid
from .train import batch_processor, set_random_seed
from sklearn import metrics
from sklearn import manifold
import matplotlib.pyplot as plt
import io
import PIL.Image
import copy
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def gen_plot(Y,Y_source,color,color_source,true_color):
    """Create a pyplot plot and save to buffer."""
    true_color=np.array(true_color)
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 8))
    cmap = plt.cm.get_cmap('jet')
    unique_colors = np.unique(np.array(color))
    axes[0].set_title("pseudo labels")
    axes[2].set_title("true labels")
    axes[3].set_title("pseudo labels & source labels")
    axes[1].set_title("source labels")
    for i,color_ in enumerate(unique_colors):
        indices = np.where(np.array(color)==color_)
        Y_ = np.array(Y)[indices[0]]
        axes[0].scatter(Y_[:, 0], Y_[:, 1], c=cmap(i/len(unique_colors)), label=color_,s=2)
        axes[3].scatter(Y_[:, 0], Y_[:, 1], c=cmap(i/len(unique_colors)), label=color_,s=2)
        print(np.mean(Y_[:, 0]),Y_[:, 0].shape)
    unique_source_color = np.unique(np.array(color_source))
    print(unique_source_color)
    for j,color_ in enumerate(unique_source_color):
        indices = np.where(np.array(color_source)==color_)
        Y_ = np.array(Y_source)[indices[0]]
        axes[1].scatter(Y_[:, 0], Y_[:, 1], c=cmap(j/len(unique_source_color)), label=color_,s=2)
        axes[3].scatter(Y_[:, 0], Y_[:, 1], color='red', label=color_,s=2)
    unique_true_colors = np.unique(true_color)
    for i,color_ in enumerate(unique_true_colors):
        indices = np.where(np.array(true_color)==color_)
        Y_ = np.array(Y)[indices[0]]
        axes[2].scatter(Y_[:, 0], Y_[:, 1], c=cmap(i/len(unique_true_colors)), label=color_,s=2)
        print(np.mean(Y_[:, 0]),Y_[:, 0].shape)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

class BaseRunner(object):
    """
    Re-ID Base Runner
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        writer,
        train_loader_similarities=None,
        train_sets=None,
        train_sets_similarities=None,
        lr_scheduler=None,
        meter_formats=None,
        print_freq=10,
        reset_optim=True,
        label_generator=None,
    ):
        super(BaseRunner, self).__init__()

        if meter_formats is None:
            meter_formats = {"Time": ":.3f", "Acc@1": ":.2%"}

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterions = criterions
        self.lr_scheduler = lr_scheduler
        self.print_freq = print_freq
        self.reset_optim = reset_optim
        self.label_generator = label_generator

        self.is_pseudo = (
            "PSEUDO_LABELS" in self.cfg.TRAIN
            and self.cfg.TRAIN.unsup_dataset_indexes is not None
        )
        if self.is_pseudo:
            if self.label_generator is None:
                self.label_generator = LabelGenerator(self.cfg, self.model) 

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0
        self._best_mAP = 0.0
        self.losses = []
        self.accs = []
        self.writer = writer
        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets
        self.train_loader_similarities, self.train_sets_similarities = train_loader_similarities, train_sets_similarities

        if "val_dataset" in self.cfg.TRAIN:
            self.val_loader, self.val_set = build_val_dataloader(cfg)
        
        # save training variables
        for key in criterions.keys():
            meter_formats[key] = ":.3f"
        self.train_progress = Meters(
            meter_formats, self.cfg.TRAIN.iters, prefix="Train: "
        )

        # build mixed precision scaler
        if "amp" in cfg.TRAIN:
            global amp_support
            if cfg.TRAIN.amp and amp_support:
                assert not isinstance(model, DataParallel), \
                    "We do not support mixed precision training with DataParallel currently"
                self.scaler = GradScaler()
            else:
                if cfg.TRAIN.amp:
                    warnings.warn(
                        "Please update the PyTorch version (>=1.6) to support mixed precision training"
                    )
                self.scaler = None
        else:
            self.scaler = None
        
        self.y_true =  [[]]
    def run(self):
        # the whole process for training
        for ep in range(self._start_epoch, self.cfg.TRAIN.epochs):
            self._epoch = ep
            # generate pseudo labels
            if self.is_pseudo:
                if (
                    ep % self.cfg.TRAIN.PSEUDO_LABELS.freq == 0
                    or ep == self._start_epoch
                ):
                    self.update_labels()
                    synchronize()
            # train
            self.train()
            synchronize()

            # validate
            
            if (ep + 1) % self.cfg.TRAIN.val_freq == 0 or (
                ep + 1
            ) == self.cfg.TRAIN.epochs:
                if "val_dataset" in self.cfg.TRAIN:
                    mAP = self.val()
                    self.save(mAP)
                else:
                    self.save()

            # update learning rate
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, list):
                    for scheduler in self.lr_scheduler:
                        scheduler.step()
                elif isinstance(self.lr_scheduler, dict):
                    for key in self.lr_scheduler.keys():
                        self.lr_scheduler[key].step()
                else:
                    self.lr_scheduler.step()

            # synchronize distributed processes
            synchronize()
        
            

    def update_labels(self):
        sep = "*************************"
        print(f"\n{sep} Start updating pseudo labels on epoch {self._epoch} {sep}\n")
        # generate pseudo labels
        pseudo_labels, label_centers,_, _ = self.label_generator(
            self._epoch, print_freq=self.print_freq
        )
        data_loaders, datasets = self.label_generator.data_loaders,self.label_generator.datasets

        tsne = False
        if tsne:
            tsne = manifold.TSNE(n_components=2, init='pca',
                                         random_state=0)

            features = tsne.fit_transform(torch.cat((all_features,all_features_source),0))
            Y = features[:len(all_features)]
            Y_source = features[len(all_features):]
            plot = gen_plot(Y,Y_source,pseudo_labels[0],list_clusters,self.y_true[int(self.cfg.task_id)])
            image = PIL.Image.open(plot)
            image = ToTensor()(image)
        #plot images of the clusters
        array_pseudo_labels = np.array(pseudo_labels[0])
        mapping = {}
        for i in np.unique(array_pseudo_labels):
            mapping[i] = np.where(array_pseudo_labels == i)[0]  
        for key in mapping.keys():
            images_to_show = []
            for n,index in enumerate(mapping[key]):
                try:
                    image = self.train_sets[0]._get_single_item(index,for_sim=True)["img"]
                    images_to_show.append(image)
                except:
                    pass
                if n >20:
                    break
            try:
                images_to_show = torchvision.utils.make_grid(images_to_show)
            except:
                pass


        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
            self.cfg,None,None,False, pseudo_labels, self.train_sets, self._epoch,
        )

        # update criterions
        if "cross_entropy" in self.criterions.keys():
            self.criterions[
                "cross_entropy"
            ].num_classes = self.train_loader.loader.dataset.num_pids

        # reset optim (optional)
        if self.reset_optim:
            self.optimizer.state = collections.defaultdict(dict)

        # update classifier centers
        start_cls_id = 0
        for idx in range(len(self.cfg.TRAIN.datasets)):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                labels = torch.arange(
                    start_cls_id, start_cls_id + self.train_sets[idx].num_pids
                )
                centers = label_centers[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
                if isinstance(self.model, list):
                    for model in self.model:
                        if isinstance(model, (DataParallel, DistributedDataParallel)):
                            model = model.module
                        model.initialize_centers(centers, labels)
                else:
                    model = self.model
                    if isinstance(model, (DataParallel, DistributedDataParallel)):
                        model = model.module
                    model.initialize_centers(centers, labels)
            start_cls_id += self.train_sets[idx].num_pids

        print(f"\n{sep} Finished updating pseudo label {sep}n")

    def train(self):
        # one loop for training
        if isinstance(self.model, list):
            for model in self.model:
                model.train()
        elif isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        self.train_progress.reset(prefix="Epoch: [{}]".format(self._epoch))

        if isinstance(self.train_loader, list):
            for loader in self.train_loader:
                loader.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)
        
        if self.train_loader_similarities is not None:
            if isinstance(self.train_loader_similarities, list):
                for loader in self.train_loader_similarities:
                    loader.new_epoch(self._epoch)
            else:
                self.train_loader_similarities.new_epoch(self._epoch)


        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):
            if isinstance(self.train_loader, list):
                batch = [loader.next() for loader in self.train_loader]
            else:
                batch = self.train_loader.next()
            if self.train_loader_similarities is not None:
                if isinstance(self.train_loader_similarities, list):
                    batch_similarities = [loader.next() for loader in self.train_loader_similarities]
                else:
                    batch_similarities = self.train_loader_similarities.next()

            if self.scaler is None:
                with torch.autograd.set_detect_anomaly(True):
                    if self.train_loader_similarities is None:
                        loss = self.train_step(iter, batch)
                    else:
                        loss = self.train_step(iter, batch, batch_similarities)
                    if (loss > 0):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

            else:
                with autocast():
                    loss = self.train_step(iter, batch)
                if (loss > 0):
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)

            if self.scaler is not None:
                self.scaler.update()

            self.train_progress.update({"Time": time.time() - end})
            end = time.time()

            if iter % self.print_freq == 0:
                self.train_progress.display(iter)

    def train_step(self, iter, batch):
        # need to be re-written case by case
        assert not isinstance(
            self.model, list
        ), "please re-write 'train_step()' to support list of models"

        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        if len(data["img"]) > 1:
            warnings.warn(
                "please re-write the 'runner.train_step()' function to make  use of "
                "mutual transformer."
            )

        inputs = data["img"][0].cuda()
        targets = data["id"].cuda()

        results = self.model(inputs)
        if "prob" in results.keys():
            results["prob"] = results["prob"][
                :, : self.train_loader.loader.dataset.num_pids
            ]

        total_loss = 0
        meters = {}
        for key in self.criterions.keys():
            if key == "cross_entropy":
                loss = self.criterions[key](results, targets)[0]
            else:
                loss = self.criterions[key](results, targets)
            if iter==self.cfg.TRAIN.iters-1:
                if self._rank == 0:
                    self.writer.add_scalar('loss_train/'+key,loss,self._epoch)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()

        self.losses.append(total_loss)
       
        acc_1 = accuracy(results["prob"].data, targets.data)
        meters["Acc@1"] = acc_1[0] 
        self.accs.append(meters["Acc@1"])
        if (iter==self.cfg.TRAIN.iters-1) & (self._rank==0):
            t_loss = sum(self.losses)/len(self.losses)
            acc_ = sum(self.accs)/len(self.accs)
            self.writer.add_scalar("TRAIN/Acc",acc_,self._epoch)
            self.writer.add_scalar("TRAIN/Total_loss",t_loss,self._epoch)
            self.losses = []
            self.accs = []

        self.train_progress.update(meters)

        return total_loss

    def val(self):

        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        better_mAP = 0
        for idx in range(len(model_list)):
            if len(model_list) > 1:
                print("==> Val on the no.{} model".format(idx))
            cmc, mAP,features,ids = val_reid(
                self.cfg,
                model_list[idx],
                self.val_loader[0],
                self.val_set[0],
                self._epoch,
                self.cfg.TRAIN.val_dataset,
                self._rank,
                print_freq=self.print_freq,
            )
            print("val map : {}".format(mAP))
            better_mAP = max(better_mAP, mAP)

            if self._rank == 0:
                print("in rank 0 val")
                self.writer.add_scalar("VAL_mAP/"+str(idx),torch.tensor(mAP,dtype = torch.float64),self._epoch)

        return better_mAP

    def save(self, mAP=None):
        if mAP is not None:
            is_best = mAP > self._best_mAP
            self._best_mAP = max(self._best_mAP, mAP)
            print(
                bcolors.OKGREEN
                + "\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n".format(
                    self._epoch, mAP, self._best_mAP, " *" if is_best else ""
                )
                + bcolors.ENDC
            )
        else:
            is_best = True
            print(
                bcolors.OKGREEN
                + "\n * Finished epoch {:3d} \n".format(self._epoch)
                + bcolors.ENDC
            )

        if self._rank == 0:
            # only on cuda:0
            self.save_model(True, self.cfg.work_dir) #is_best=True

    def save_model(self, is_best, fpath):
        if isinstance(self.model, list):
            state_dict = {}
            state_dict["epoch"] = self._epoch + 1
            state_dict["best_mAP"] = self._best_mAP
            for idx, model in enumerate(self.model):
                state_dict["state_dict_" + str(idx + 1)] = model.state_dict()
            save_checkpoint(state_dict, is_best,self.cfg.task_id,
                    fpath=osp.join(fpath, "checkpoint.pth"),fpath_id=osp.join(fpath, "checkpoint.pth"))

        elif isinstance(self.model, dict):
            state_dict = {}
            state_dict["epoch"] = self._epoch + 1
            state_dict["best_mAP"] = self._best_mAP
            for key in self.model.keys():
                state_dict["state_dict"] = self.model[key].state_dict()
                save_checkpoint(state_dict, is_best,self.cfg.task_id,
                        fpath=osp.join(fpath, "checkpoint.pth"),fpath_id=osp.join(fpath, "checkpoint.pth"))

        elif isinstance(self.model, nn.Module):
            state_dict = {}
            state_dict["epoch"] = self._epoch + 1
            state_dict["best_mAP"] = self._best_mAP
            state_dict["state_dict"] = self.model.state_dict()
            save_checkpoint(state_dict, is_best,self.cfg.task_id,
                        fpath=osp.join(fpath, "checkpoint.pth"),fpath_id=osp.join(fpath, "checkpoint.pth"))

        else:
            assert "Unknown type of model for save_model()"
    def resume_bis(self,path):
        self.load_model_bis(path)
        synchronize()
    def load_model_bis(self,path):
        state_dict = load_checkpoint(path)
        self._start_epoch = state_dict["epoch"]
        self._best_mAP = state_dict["best_mAP"]
    def resume(self, path):
        # resume from a training checkpoint (not source pretrain)
        self.load_model(path)
        synchronize()

    def load_model(self, path):
        if isinstance(self.model, list):
            print("model is list of models")
            assert osp.isfile(path)
            state_dict = load_checkpoint(path)
            for idx, model in enumerate(self.model):
                copy_state_dict(state_dict["state_dict_" + str(idx + 1)], model)

        elif isinstance(self.model, dict):
            assert osp.isdir(path)
            for key in self.model.keys():
                state_dict = load_checkpoint(osp.join(path, "checkpoint_"+key+".pth"))
                copy_state_dict(state_dict["state_dict"], self.model[key])

        elif isinstance(self.model, nn.Module):
            assert osp.isfile(path)
            state_dict = load_checkpoint(path)
            if "state_dict" in state_dict.keys():
                copy_state_dict(state_dict["state_dict"], self.model)
            elif "state_dict_1" in state_dict.keys():
                copy_state_dict(state_dict["state_dict_1"], self.model)

        self._start_epoch = state_dict["epoch"]
        self._best_mAP = state_dict["best_mAP"]
        self.model_teacher = copy.deepcopy(self.model)

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size


class GANBaseRunner(BaseRunner):
    """
    Domain-translation Base Runner
    Re-implementation of CycleGAN
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        writer,
        **kwargs
    ):
        super(GANBaseRunner, self).__init__(
            cfg, model, optimizer, criterions, train_loader,writer, **kwargs
        )

        self.save_dir = osp.join(self.cfg.work_dir, 'images')
        if self._rank == 0:
            mkdir_if_missing(self.save_dir)

        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()

    def train_step(self, iter, batch):
        data_src, data_tgt = batch[0], batch[1]

        self.real_A = data_src['img'].cuda()
        self.real_B = data_tgt['img'].cuda()

        # Forward
        self.fake_B = self.model['G_A'](self.real_A)     # G_A(A)
        self.fake_A = self.model['G_B'](self.real_B)     # G_B(B)
        self.rec_A = self.model['G_B'](self.fake_B)    # G_B(G_A(A))
        self.rec_B = self.model['G_A'](self.fake_A)    # G_A(G_B(B))

        # G_A and G_B
        self.set_requires_grad([self.model['D_A'], self.model['D_B']], False) # save memory
        if self.scaler is None:
            self.optimizer['G'].zero_grad()
        else:
            with autocast(enabled=False):
                self.optimizer['G'].zero_grad()
        self.backward_G()
        if self.scaler is None:
            self.optimizer['G'].step()
        else:
            with autocast(enabled=False):
                self.scaler.step(self.optimizer['G'])

        # D_A and D_B
        self.set_requires_grad([self.model['D_A'], self.model['D_B']], True)
        if self.scaler is None:
            self.optimizer['D'].zero_grad()
        else:
            with autocast(enabled=False):
                self.optimizer['D'].zero_grad()
        self.backward_D()
        if self.scaler is None:
            self.optimizer['D'].step()
        else:
            with autocast(enabled=False):
                self.scaler.step(self.optimizer['D'])

        # save translated images
        if self._rank == 0:
            self.save_imgs(['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B'])

        return 0

    def backward_G(self, retain_graph=False):
        """Calculate the loss for generators G_A and G_B"""
        # Adversarial loss D_A(G_A(B))
        loss_G_A = self.criterions['gan_G'](self.model['D_A'](self.fake_A), True)
        # Adversarial loss D_B(G_B(A))
        loss_G_B = self.criterions['gan_G'](self.model['D_B'](self.fake_B), True)
        loss_G = loss_G_A + loss_G_B

        # Forward cycle loss || G_A(G_B(A)) - A||
        loss_recon_A = self.criterions['recon'](self.rec_A, self.real_A)
        # Backward cycle loss || G_B(G_A(B)) - B||
        loss_recon_B = self.criterions['recon'](self.rec_B, self.real_B)
        loss_recon = loss_recon_A + loss_recon_B

        # G_A should be identity if real_B is fed: ||G_B(B) - B||
        idt_A = self.model['G_A'](self.real_B)
        loss_idt_A = self.criterions['ide'](idt_A, self.real_B)
        # G_B should be identity if real_A is fed: ||G_A(A) - A||
        idt_B = self.model['G_B'](self.real_A)
        loss_idt_B = self.criterions['ide'](idt_B, self.real_A)
        loss_idt = loss_idt_A + loss_idt_B

        # combined loss and calculate gradients
        loss = loss_G * self.cfg.TRAIN.LOSS.losses['gan_G'] + \
                loss_recon * self.cfg.TRAIN.LOSS.losses['recon'] + \
                loss_idt * self.cfg.TRAIN.LOSS.losses['ide']
        if self.scaler is None:
            loss.backward(retain_graph=retain_graph)
        else:
            with autocast(enabled=False):
                self.scaler.scale(loss).backward(retain_graph=retain_graph)

        meters = {'gan_G': loss_G.item(),
                  'recon': loss_recon.item(),
                  'ide': loss_idt.item()}
        self.train_progress.update(meters)

    def backward_D_basic(self, netD, real, fake, fake_pool):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterions['gan_D'](pred_real, True)
        # Fake
        pred_fake = netD(fake_pool.query(fake))
        loss_D_fake = self.criterions['gan_D'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss = loss_D * self.cfg.TRAIN.LOSS.losses['gan_D']
        if self.scaler is None:
            loss.backward()
        else:
            with autocast(enabled=False):
                self.scaler.scale(loss).backward()
        return loss_D.item()

    def backward_D(self):
        loss_D_A = self.backward_D_basic(self.model['D_A'], self.real_A, self.fake_A, self.fake_A_pool)
        loss_D_B = self.backward_D_basic(self.model['D_B'], self.real_B, self.fake_B, self.fake_B_pool)
        meters = {'gan_D': loss_D_A + loss_D_B}
        self.train_progress.update(meters)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_imgs(self, names):
        for name in names:
            img = getattr(self, name)[0]
            img_np = tensor2im(img, mean=self.cfg.DATA.norm_mean, std=self.cfg.DATA.norm_std)
            save_image(img_np, osp.join(self.save_dir, 'epoch_{}_{}.jpg'.format(self._epoch, name)))
