import os
from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
import tensorboard as tb
import matplotlib.pyplot as plt
from sklearn import manifold
import io
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
import argparse
import collections
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch, torchvision
from torch.nn.parallel import DataParallel, DistributedDataParallel

from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import (
    build_test_dataloader,
    build_train_dataloader,
    build_val_dataloader,
)
from openunreid.core.metrics.accuracy import accuracy
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.models.utils.extract import extract_features, extract_features_for_similarities
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize, get_dist_info
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from openunreid.utils.torch_utils import tensor2im
from openunreid.data.utils.data_utils import save_image
n_tasks = 5
def gen_plot(Y,color):
    """Create a pyplot plot and save to buffer."""
    plt.figure(figsize=(15,8))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
def filtre_source_(train_sets,features,tag,writer,rank, save_images=False):
    target = train_sets[0]
    source = train_sets[1]
    cos = torch.nn.CosineSimilarity(dim=-1)
    features_target = features["target"]
    features_source = features["source"]
    id_to_indice_t = {id:indice for id,indice in enumerate(features_target.keys())}
    id_to_indice_s = {id:indice for id,indice in enumerate(features_source.keys())}
    similarities = torch.zeros(len(id_to_indice_t),len(id_to_indice_s))
    for i in range(len(id_to_indice_t.keys())):
        for j in range(len(features_source)):
            similarities[i,j]= cos(features_target[id_to_indice_t[list(id_to_indice_t.keys())[i]]],features_source[id_to_indice_s[j]])
    indices_min_source = similarities.max(axis=1)[1]
    similarities_min_source = similarities.max(axis=1)[0]
    fst_quantile = torch.quantile(torch.tensor(similarities.max(axis=1)[0]),0.25)
    trd_quantile = torch.quantile(torch.tensor(similarities.max(axis=1)[0]),0.75)
    target_similar_source = {}
    target_images = []
    source_images = []
    images_to_show_fstq = []
    images_to_show_trdq = []
    source_image_ids = []
    source_image_paths = []
    for i in range(len(indices_min_source)):
        target_similar_source[id_to_indice_t[list(id_to_indice_t.keys())[i]]]=id_to_indice_s[indices_min_source[i].item()]
        target_image = target._get_single_item(id_to_indice_t[list(id_to_indice_t.keys())[i]],for_sim=True)["img"]
        source_image = source._get_single_item(id_to_indice_s[indices_min_source[i].item()],for_sim=True)["img"]
        source_image_id = source._get_single_item(id_to_indice_s[indices_min_source[i].item()],for_sim=True)["id"]
        source_image_path = source._get_single_item(id_to_indice_s[indices_min_source[i].item()],for_sim=True)["path"]
        if similarities_min_source[i]<=fst_quantile:
            image_to_show_fstq = torchvision.utils.make_grid([target_image,source_image])
            images_to_show_fstq.append(image_to_show_fstq)

        elif similarities_min_source[i]>=trd_quantile:
            image_to_show_trdq = torchvision.utils.make_grid([target_image,source_image])
            images_to_show_trdq.append(image_to_show_trdq)
            if save_images:
                directory = os.path.join(cfg.work_dir, "similar_images_support/")
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path_s = os.path.join(cfg.work_dir, "similar_images_support/image_source_" + str(i)+'.png')
                save_path_t = os.path.join(cfg.work_dir, "similar_images_support/image_target_" + str(i)+'.png')
                torchvision.utils.save_image(target_image,save_path_t)
                torchvision.utils.save_image(source_image,save_path_s)

        if source_image_id not in source_image_ids:
            source_image_ids.append(source_image_id)
        if source_image_path not in source_image_paths:
            source_image_paths.append(source_image_path)
            
        # except:
        #     continue
    images_to_show_trdq = torchvision.utils.make_grid(images_to_show_trdq)
    images_to_show_fstq = torchvision.utils.make_grid(images_to_show_fstq)
    if rank == 0:
        writer.add_image(tag + ' images target / source 1st q', images_to_show_fstq, 0)
        writer.add_image(tag + ' images target / source 3rd q', images_to_show_trdq, 0)
        print("done images / similarities")
    return similarities,target_similar_source,source_image_ids,source_image_paths
class SpCLRunner(BaseRunner):
    def update_labels(self):
        sep = "*************************"
        print(f"\n{sep} Start updating pseudo labels on epoch {self._epoch} {sep}\n")

        memory_features = []
        start_ind = 0
        for idx, dataset in enumerate(self.train_sets):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                memory_features.append(
                    self.criterions["hybrid_memory"]
                    .features[start_ind : start_ind + len(dataset)]
                    .clone()
                    .cpu()
                )
                start_ind += len(dataset)
            else:
                start_ind += dataset.num_pids

        # generate pseudo labels
        pseudo_labels, label_centers ,_ ,_= self.label_generator(
            self._epoch, memory_features=memory_features, print_freq=self.print_freq
        )

        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
            self.cfg, None, None, False, pseudo_labels, self.train_sets, self._epoch, joint=False
        )

        # update memory labels
        memory_labels = []
        memory_labels_target = []
        memory_labels_source = []
        start_pid = 0
        start_pid_target = 0
        start_pid_source = 0
        start_pid = 0
        for idx, dataset in enumerate(self.train_sets):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                labels = pseudo_labels[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
                memory_labels.append(torch.LongTensor(labels) + start_pid)
                memory_labels_target.append(torch.LongTensor(labels) + start_pid_target)
                start_pid += max(labels) + 1
                start_pid_target += max(labels) + 1
            else:
                num_pids = dataset.num_pids
                memory_labels.append(torch.arange(start_pid, start_pid + num_pids))
                memory_labels_source.append(torch.arange(start_pid_source, start_pid_source + num_pids))
                start_pid += num_pids
                start_pid_source += num_pids
        memory_labels = torch.cat(memory_labels).view(-1)
        memory_labels_target = torch.cat(memory_labels_target).view(-1)
        memory_labels_source = torch.cat(memory_labels_source).view(-1)
        self.criterions["hybrid_memory"]._update_label(memory_labels, memory_labels_target, memory_labels_source)

        print(f"\n{sep} Finished updating pseudo label {sep}\n")

    def train_step(self, iter, batch,batch_similarities=None, multi_loader_similarities=True):
        task_id = self.cfg.task_id
        start_ind, start_pid = 0, 0
        for idx, sub_batch in enumerate(batch):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                sub_batch["ind"] += start_ind
                start_ind += len(self.train_sets[idx])
            else:
                sub_batch["ind"] = sub_batch["id"] + start_ind
                start_ind += self.train_sets[idx].num_pids

            sub_batch["id"] += start_pid
            start_pid += self.train_sets[idx].num_pids
        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        len_ = int(len(data["id"])/2) 
        if batch_similarities is not None:
            if multi_loader_similarities:
                data_similarities = [batch_processor(batch_similarities_, self.cfg.MODEL.dsbn) for batch_similarities_ in batch_similarities]
                inputs_similarities = [data_similarities_["img"][0].cuda() for data_similarities_ in data_similarities]          
            else:
                data_similarities = batch_processor(batch_similarities, self.cfg.MODEL.dsbn)
                inputs_similarities = [data_similarities["img"][0].cuda()]
        else:
            inputs_similarities = torch.tensor([]).cuda()

        inputs = data["img"][0].cuda()
        targets = data["id"].cuda()
        indexes = data["ind"].cuda()
        lens = [len(inputs_sim) for inputs_sim in inputs_similarities]
        lens.insert(0,len(inputs))
        for i in range (len(lens)):
            for j in range(i):
                lens[i]+=lens[j]
        if len(inputs_similarities)>1:
            inputs_similarities = torch.cat(inputs_similarities)
        elif len(inputs_similarities)==1:
            inputs_similarities = inputs_similarities[0]

        # results = self.model(inputs)
        inputs_mo = torch.cat((inputs,inputs_similarities))
        results, results_mean = self.model(inputs_mo)

        if self.cfg.KD=="S2P":
            results_student = [results["feat"][lens[i]:lens[i+1]] for i in range(len(lens)-1)] #for kd optimization   
        elif self.cfg.KD=="SP":
            results_student = [results["activation"][lens[i]:lens[i+1]] for i in range(len(lens)-1)] 
        elif self.cfg.KD=="LwF":
            results_student = [results["activation"][lens[i]:lens[i+1]] for i in range(len(lens)-1)]   
        
        results_target , results_source = results["feat"][:len_], results_mean["feat"][len_:2*len_] #for mmd optimization
        if task_id>0:
            with torch.no_grad():
                    if self.cfg.KD=="S2P":
                        results_teacher = [results_mean["feat"][lens[i]:lens[i+1]] for i in range(len(lens)-1)] 
                    elif self.cfg.KD=="SP":
                        results_teacher = [results_mean["activation"][lens[i]:lens[i+1]] for i in range(len(lens)-1)] 
                    elif self.cfg.KD=="LwF":
                        results_teacher = [results_mean["activation"][lens[i]:lens[i+1]] for i in range(len(lens)-1)] 
                    # results_source = results_mean["feat"][2*len_:] #for mmd optimization/ switch to dream
                
        total_loss = 0
        for key in results.keys():
            results[key]=results[key][:len(inputs)]
        for key in results.keys():
            results_mean[key]=results_mean[key][:len(inputs)]
        meters = {}

        for key in self.criterions.keys():
            if key == "hybrid_memory":
                loss = self.criterions[key](results, indexes) 
            elif key =="MMD_loss":
                loss = self.criterions[key](results_source,results_target)
                # print("loss_mmd {}".format(loss))
            elif key == "kd_loss" or key == "KDLossAT" or key == "KDLossSP":

                if task_id > 0:
                    results_student = torch.cat(results_student)
                    results_teacher = torch.cat(results_teacher)
                    loss = self.criterions[key](results_student,results_teacher) #/len(results_student)   #Mean on the list of dataloaders

                else:
                    loss = loss*0
            if iter==self.cfg.TRAIN.iters-1:
                if self._rank == 0:
                    self.writer.add_scalar('loss_train/'+key,loss,self._epoch)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()
        self.losses.append(total_loss)
        if (iter==self.cfg.TRAIN.iters-1) & (self._rank==0):
            t_loss = sum(self.losses)/len(self.losses)
            self.writer.add_scalar("train/Total_loss",t_loss,self._epoch)
            self.losses = []
        self.train_progress.update(meters)
        
        return total_loss


def parge_config():
    parser = argparse.ArgumentParser(description="SpCL training")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models", default=""
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--KDloss", type=float, default=0.)
    parser.add_argument("--KD", type=str, default="S2P") # LwF or SP or S2P
    parser.add_argument("--MMDloss", type=float, default=0.)
    parser.add_argument("--exec", type=float) #if 0 then local, 1 fastml, 2 Telecom Cluster
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()
    cfg_from_yaml_file(args.config, cfg)
    assert cfg.TRAIN.PSEUDO_LABELS.use_outliers
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg.TRAIN.epochs = args.epochs
    cfg.TRAIN.iters = args.iters
    cfg.KD = args.KD
    # cfg.TRAIN.seed = args.seed
    if cfg.KD == "LwF":
        cfg.TRAIN.LOSS.losses["KDLossAT"] = cfg.TRAIN.LOSS.losses.pop("kd_loss")
        if args.KDloss >0:
            cfg.TRAIN.LOSS.losses["KDLossAT"] = args.KDloss
    elif cfg.KD == "SP":
        cfg.TRAIN.LOSS.losses["KDLossSP"] = cfg.TRAIN.LOSS.losses.pop("kd_loss")
        if args.KDloss >0:
            cfg.TRAIN.LOSS.losses["KDLossSP"] = args.KDloss
    elif cfg.KD == "S2P":
        if args.KDloss >0:
            cfg.TRAIN.LOSS.losses["kd_loss"] = args.KDloss

    if args.MMDloss>0:
        cfg.TRAIN.LOSS.losses["MMD_loss"] = args.MMDloss
    cfg.exec = args.exec
    if cfg.exec==0:
        print("Training locally on naboo")
        cfg.DATA_ROOT = Path(cfg.DATA_ROOT_local)
        cfg.LOGS_ROOT = Path(cfg.LOGS_ROOT_local)
        cfg.MODEL.backbone_path = cfg.MODEL.backbone_path_local
        cfg.MODEL.source_pretrained = cfg.MODEL.source_pretrained_local
    elif cfg.exec==2:
        print("training on slurm Cluster")
        cfg.DATA_ROOT = Path(cfg.DATA_ROOT_cluster)
        cfg.LOGS_ROOT = Path(cfg.LOGS_ROOT_cluster)
        cfg.MODEL.backbone_path = cfg.MODEL.backbone_path_cluster
        cfg.MODEL.source_pretrained = cfg.MODEL.source_pretrained_cluster
    else:
        print("training on fastml")
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    mkdir_if_missing(cfg.work_dir)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    rank,_,_ = get_dist_info()
    if rank == 0:
        if cfg.exec==0:
            writer = SummaryWriter(cfg.work_dir / 'logs_tb/',flush_secs=10)
        elif cfg.exec==2:
            writer = SummaryWriter(cfg.work_dir / 'logs_tb/',flush_secs=10)
        else:
            writer = SummaryWriter('/out/logs_tb/',flush_secs=10)
    else :
        writer = None
    synchronize()
    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)
    cfg.n_tasks = n_tasks
    cfg.task_id=0
    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg, n_tasks, 0, [True,False], joint=False)
    # build model
    model = build_model(cfg, 0, [], init=cfg.MODEL.source_pretrained)
    model.cuda()
    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        model = DistributedDataParallel(model, **ddp_cfg)
    elif cfg.total_gpus > 1:
        model = DataParallel(model)

    # build optimizer
    optimizer = build_optimizer([model], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None
    # build loss functions
    num_memory = 0
    num_memory_target = 0
    num_memory_source = 0
    for idx, set in enumerate(train_sets):
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # instance-level memory for unlabeled data
            num_memory += len(set)
            num_memory_target += len(set)
        else:
            # class-level memory for labeled data
            num_memory += set.num_pids
            num_memory_source += set.num_pids

    if isinstance(model, (DataParallel, DistributedDataParallel)):
        num_features = model.module.net.num_features
    else:
        num_features = model.num_features

    criterions = build_loss(
        cfg.TRAIN.LOSS,
        num_features=num_features,
        num_memory=num_memory,
        num_memory_target=num_memory_target,
        num_memory_source=num_memory_source,
        cuda=True,
    )

    # init memory
    loaders, datasets = build_val_dataloader(
        cfg, for_clustering=True, all_datasets=True, SpCL=True
    )
    memory_features = []
    memory_features_target = []
    memory_features_source = []
    for idx, (loader, dataset) in enumerate(zip(loaders, datasets)):
        features = extract_features(
            model, loader, dataset, with_path=False, prefix="Extract: ",
        )
        assert features.size(0) == len(dataset)
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # init memory for unlabeled data with instance features
            memory_features.append(features)
            memory_features_target.append(features)
        else:
            # init memory for labeled data with class centers
            centers_dict = collections.defaultdict(list)
            for i, (_, pid, _) in enumerate(dataset):
                centers_dict[pid].append(features[i].unsqueeze(0))
            centers = [
                torch.cat(centers_dict[pid], 0).mean(0)
                for pid in sorted(centers_dict.keys())
            ]
            memory_features.append(torch.stack(centers, 0))
            memory_features_source.append(torch.stack(centers, 0))
    del loaders, datasets
    memory_features = torch.cat(memory_features)
    memory_features_target = torch.cat(memory_features_target)
    memory_features_source = torch.cat(memory_features_source)
    criterions["hybrid_memory"]._update_feature(memory_features)

    # build runner
    runner = SpCLRunner(
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        writer,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        meter_formats={"Time": ":.3f",},
        reset_optim=False,
    )


    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    test_loaders, queries, galleries = build_test_dataloader(cfg)
    l= ["source","target"]
    # for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
    #     if cfg.TEST.datasets[i]=="msmt17" and i==0:
    #         mAP = 0
    #     else:
    #         cmc, mAP = test_reid(
    #             cfg, runner.model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i], visrankactiv=False
    #         )
    #     if rank == 0:
    #         writer.add_scalar("TEST_Map_"+l[i]+"/"+str(1),float(mAP),0)
    print("start_training task 0")
    epochs_per_task = cfg.TRAIN.epochs
    # # start training
    runner.run()
    source_image_ids_list = []
    train_loader_similarities_list = []
    for task in range(n_tasks-1):
        cfg.TRAIN.epochs+=epochs_per_task
        cfg.task_id+=1
        print("start training task {} out of {} tasks".format(task+1,n_tasks))
        # build train loader
        del train_loader, train_sets, model
        torch.cuda.empty_cache()
        train_loader, train_sets = build_train_dataloader(cfg,
                n_tasks, task+1,[True,False])

                # Compute similarities and dataloader similarities #

        train_loader_similarities, train_sets_similarities = build_train_dataloader(cfg, 
                n_tasks, task,[True,False],for_kd=True)
        features = extract_features_for_similarities(runner.model, train_loader_similarities, train_sets_similarities)
            
        similarities,target_similar_source,source_image_ids,source_image_paths = filtre_source_(train_sets_similarities,features,'after task '+str(task+1),writer,rank, save_images=False)
        
        color = ['b']*len(features["target"])+['g']*len(features["source"])
        features = np.array([t.numpy() for t in features["target"].values()]+[s.numpy() for s in features["source"].values()])
        
        tsne = manifold.TSNE(n_components=2, init='pca',
                                    random_state=0)
        Y = tsne.fit_transform(features)
        plot = gen_plot(Y,color)
        # print(Y)
        image = PIL.Image.open(plot)
        image = ToTensor()(image)
        if rank == 0:
            writer.add_image('tsne after task {}'.format(task+1),image,0)


        source_image_ids_list+= source_image_ids
        train_loader_similarities, train_sets_similarities = build_train_dataloader(cfg, 
                n_tasks, task,[True,source_image_ids],for_kd=True,only_source=True)
        train_loader_similarities_list.append(train_loader_similarities)
                       #########################################

        # build model
        model = build_model(cfg, 0, [], init=cfg.MODEL.source_pretrained, kd_flag=True)
        model.cuda()
        if dist:
            ddp_cfg = {
                "device_ids": [cfg.gpu],
                "output_device": cfg.gpu,
                "find_unused_parameters": True,
            }
            model = DistributedDataParallel(model, **ddp_cfg)
        elif cfg.total_gpus > 1:
            model = DataParallel(model)

        # build optimizer
        optimizer = build_optimizer([model], **cfg.TRAIN.OPTIM)

        # build lr_scheduler
        if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
            lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
        else:
            lr_scheduler = None
        num_memory = 0
        num_memory_target = 0
        num_memory_source = 0
        for idx, set in enumerate(train_sets):
            if idx in cfg.TRAIN.unsup_dataset_indexes:
                # instance-level memory for unlabeled data
                num_memory += len(set)
                num_memory_target += len(set)
            else:
                # class-level memory for labeled data
                num_memory += set.num_pids
                num_memory_source += set.num_pids
        print("num memory : {}".format(num_memory))
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            num_features = model.module.net.num_features
        else:
            num_features = model.num_features
        print("num_features : {}".format(num_features))
        criterions = build_loss(
            cfg.TRAIN.LOSS,
            num_features=num_features,
            num_memory=num_memory,
            num_memory_target=num_memory_target,
            num_memory_source=num_memory_source,
            cuda=True,
        ) 
        # init memory
        loaders, datasets = build_val_dataloader(
            cfg, for_clustering=True, all_datasets=True, SpCL=True
        )
        memory_features = []
        memory_features_target = []
        memory_features_source = []
        for idx, (loader, dataset) in enumerate(zip(loaders, datasets)):
            features = extract_features(
                model, loader, dataset, with_path=False, prefix="Extract: ",
            )
            assert features.size(0) == len(dataset)
            if idx in cfg.TRAIN.unsup_dataset_indexes:
                # init memory for unlabeled data with instance features
                memory_features.append(features)
                memory_features_target.append(features)
            else:
                # init memory for labeled data with class centers
                centers_dict = collections.defaultdict(list)
                for i, (_, pid, _) in enumerate(dataset):
                    centers_dict[pid].append(features[i].unsqueeze(0))
                centers = [
                    torch.cat(centers_dict[pid], 0).mean(0)
                    for pid in sorted(centers_dict.keys())
                ]
                memory_features.append(torch.stack(centers, 0))
                memory_features_source.append(torch.stack(centers, 0))
        del loaders, datasets
        memory_features = torch.cat(memory_features)
        memory_features_target = torch.cat(memory_features_target)
        memory_features_source = torch.cat(memory_features_source)
        criterions["hybrid_memory"]._update_feature(memory_features)
        del runner

        runner = SpCLRunner(
         cfg,
         model,
         optimizer,
         criterions,
         train_loader,
         writer,
         train_loader_similarities = train_loader_similarities,
         train_sets=train_sets,
         lr_scheduler=lr_scheduler,
         meter_formats={"Time": ":.3f",},
         reset_optim=False,
          )
        runner.resume(cfg.work_dir / "model_best.pth")
        for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
            if cfg.TEST.datasets[i]=="msmt17" and i==0:
                mAP = 0
            else:
                cmc, mAP = test_reid(
                    cfg, runner.model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i], visrankactiv=False
                )
            if rank == 0:
                writer.add_scalar("TEST_Map_"+l[i]+"/"+str(1),float(mAP),task+1)
        runner.run()


    # load the best model
    #resume important !
    runner.resume(cfg.work_dir / "model_best.pth")
    # final testing
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        if cfg.TEST.datasets[i]=="msmt17" and i==0:
            mAP = 0
        else:
            cmc, mAP = test_reid(
                cfg, runner.model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i], visrankactiv=False
            )
            if cfg.exec==1:
                    f = open("/optim/score.txt", 'w')
                    f.write('%g' % float(mAP))
                    f.close()
        if rank == 0:
                writer.add_scalar("TEST_Map_"+l[i]+"/"+str(1),float(mAP),n_tasks)

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))

if __name__ == "__main__":
    main()