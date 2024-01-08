from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
import tensorflow as tf
from sklearn import manifold
import tensorboard as tb
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import PIL.Image
from openunreid.models.utils.extract import extract_features, extract_features_for_similarities
from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.metrics.accuracy import accuracy
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader
from openunreid.data.builder import build_train_dataloader_for_sim

from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize, get_dist_info
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
from openunreid.apis.test import val_reid
import copy
import io
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def gen_plot(Y,color):
    """Create a pyplot plot and save to buffer."""
    plt.figure(figsize=(15,8))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
def filtre_source_(train_sets,features,tag,writer,rank):
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
        try:
            # print(i)
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
            if source_image_id not in source_image_ids:
                source_image_ids.append(source_image_id)
            source_image_paths.append(source_image_path)
            
        except:
            continue
    images_to_show_trdq = torchvision.utils.make_grid(images_to_show_trdq)
    images_to_show_fstq = torchvision.utils.make_grid(images_to_show_fstq)
    if rank == 0:
        writer.add_image(tag + ' images target / source 1st q', images_to_show_fstq, 0)
        writer.add_image(tag + ' images target / source 3rd q', images_to_show_trdq, 0)
        writer.add_histogram(tag + ' similarities target / source', similarities.max(axis=1)[0], 0)
        print("done images / similarities")
    return similarities,target_similar_source,source_image_ids,source_image_paths


n_tasks = 5
class MMTRunner(BaseRunner):
    def train_step(self, iter, batch, batch_similarities=None,  multi_loader_similarities=True):
        batch_similarities=None
        task_id = self.cfg.task_id
        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        len_ = int(len(data["id"])/2)
        if batch_similarities is not None:
            data_similarities = [batch_processor(batch_similarities_, self.cfg.MODEL.dsbn) for batch_similarities_ in batch_similarities]
            inputs_similarities_1 = [data_similarities_["img"][0].cuda() for data_similarities_ in data_similarities] 
            inputs_similarities_2 = [data_similarities_["img"][1].cuda() for data_similarities_ in data_similarities] 
        else:
            inputs_similarities_1 = torch.tensor([]).cuda()
            inputs_similarities_2 = torch.tensor([]).cuda()
        
        inputs_1 = data["img"][0].cuda()
        inputs_2 = data["img"][1].cuda()

        targets = data["id"].cuda()
        inputs_mo1 = inputs_1 
        inputs_mo2 = inputs_2 

        results_1, results_1_mean = self.model[0](inputs_mo1)

        results_2, results_2_mean = self.model[1](inputs_mo2)
    
        results_1["prob"] = results_1["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]
        
        results_1_mean["prob"] = results_1_mean["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]


        results_2["prob"] = results_2["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]
        results_2_mean["prob"] = results_2_mean["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]

        for key in results_1.keys():
            results_1[key]=results_1[key][:len(inputs_1)]
        for key in results_1_mean.keys():
            results_1_mean[key]=results_1_mean[key][:len(inputs_1)]
        for key in results_2.keys():
            results_2[key]=results_2[key][:len(inputs_2)]
        for key in results_2_mean.keys():
            results_2_mean[key]=results_2_mean[key][:len(inputs_2)]
        
        total_loss = 0
        meters = {}
        for key in self.criterions.keys():
            if key == "soft_entropy":
                loss = self.criterions[key](
                    results_1, results_2_mean
                )[0] + self.criterions[key](results_2, results_1_mean)[0]
                # loss_vec = self.criterions[key](
                    # results_1, results_2_mean
                # )[1] + self.criterions[key](results_2, results_1_mean)[1]
            elif key == "soft_softmax_triplet":
                loss = self.criterions[key](
                        results_1, targets, results_2_mean
                ) + self.criterions[key](results_2, targets, results_1_mean)
            elif key == "cross_entropy":
                loss = self.criterions[key](results_1, targets)[0] + self.criterions[key](results_2, targets)[0]
                # loss_vec = self.criterions[key](results_1, targets)[1] + self.criterions[key](results_2, targets)[1]          
            else:
                loss = self.criterions[key](results_1, targets) + self.criterions[key](results_2, targets)
            if iter==self.cfg.TRAIN.iters-1:
                if key=="soft_entropy" or key=="cross_entropy":
                    if self._rank == 0:
                        print("in rank 0 loss")
                        self.writer.add_scalar('losses/'+key,loss,self._epoch)
                else:  
                    if self._rank == 0:  
                        print("in rank 0 loss")
                        self.writer.add_scalar('losses/'+key,loss,self._epoch)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])   
            meters[key] = loss.item()
        
        self.losses.append(total_loss)
       
        acc_1 = accuracy(results_1["prob"].data, targets.data)
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
    parser.add_argument("--exec", type=float) #if 0 then local, else fastml
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()
    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg.exec = args.exec
    if cfg.exec==0:
        print("Training locally on naboo")
        cfg.DATA_ROOT = Path(cfg.DATA_ROOT_local)
        cfg.LOGS_ROOT = Path(cfg.LOGS_ROOT_local)
        cfg.MODEL.backbone_path = cfg.MODEL.backbone_path_local
        cfg.MODEL.source_pretrained = cfg.MODEL.source_pretrained_local

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
    rank,_,_ = get_dist_info()
    print(rank)
    if rank == 0:
        if cfg.exec==0:
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
    # sys.exit()
    cfg.n_tasks = n_tasks
    cfg.task_id=0
    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg, n_tasks,
            0,[True,False], joint=False)
    # the number of classes for the model is tricky,
    # you need to make sure that
    # it is always larger than the number of clusters
    num_classes = 0
    for idx, set in enumerate(train_sets):
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # number of clusters in an unsupervised dataset
            # must not be larger than the number of images
            num_classes += len(set)
        else:
            # ground-truth classes for supervised dataset
            num_classes += set.num_pids

    # build model no.1
    model_1 = build_model(cfg, num_classes,[], init=cfg.MODEL.source_pretrained)
    model_1.cuda()
    # build model no.2
    model_2 = build_model(cfg, num_classes,[], init=cfg.MODEL.source_pretrained)
    model_2.cuda()
    #print(model_1.net.classifier)
    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        model_1 = torch.nn.parallel.DistributedDataParallel(model_1, **ddp_cfg)
        model_2 = torch.nn.parallel.DistributedDataParallel(model_2, **ddp_cfg)
    elif cfg.total_gpus > 1:
        model_1 = torch.nn.DataParallel(model_1)
        model_2 = torch.nn.DataParallel(model_2)
    
    # build optimizer
    optimizer = build_optimizer([model_1, model_2], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None

    # build loss functions
    criterions = build_loss(cfg.TRAIN.LOSS, num_classes=num_classes, cuda=True)
    # print("model 1 before 1st training {}".format(model_1.module.net.parameters()))
    # build runner
    runner = MMTRunner(
        cfg,
        [model_1, model_2,],
        optimizer,
        criterions,
        train_loader,
        writer,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        reset_optim=True,
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)
    epochs_per_task = cfg.TRAIN.epochs
    test_loaders, queries, galleries = build_test_dataloader(cfg, n_tasks, 0, False)
    l= ["source","target"]
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):

        for idx in range(len(runner.model)):
            if cfg.TEST.datasets[i]=="msmt17" and i==0:
                mAP = 0
            else:

                print("==> Test on the no.{} model".format(idx))
                # test_reid() on self.model[idx] will only evaluate the 'mean_net'
                # for testing 'net', use self.model[idx].module.net
                cmc, mAP = test_reid(
                    cfg,
                    runner.model[idx],
                    loader,
                    query,
                    gallery,
                    dataset_name=cfg.TEST.datasets[i],
                    visrankactiv= False,
                )
                print("map on "+l[i]+" domain : {}".format(mAP))
            if rank == 0:
                # print("in rank 0 test")
                writer.add_scalar("TEST_Map_"+l[i]+"/"+str(idx),float(mAP),0)
    # start training
    print("start_training task 1")
    runner.run()
    for task in range(n_tasks-1):
        print("start training task {} out of {} tasks".format(task+2,n_tasks))
        # build train loader
        train_loader, train_sets = build_train_dataloader(cfg,
                n_tasks, task+1,[True,False])
        cfg.TRAIN.epochs+=epochs_per_task
        cfg.task_id+=1
        #build new models to train with previous old classifiers for kd loss
        model_1 = build_model(cfg, num_classes,[], init=cfg.MODEL.source_pretrained)
        model_1.cuda()
        # build model no.2
        model_2 = build_model(cfg, num_classes,[], init=cfg.MODEL.source_pretrained)
        model_2.cuda()
        #print(model_1.net.classifier)
        if dist:
            ddp_cfg = {
                "device_ids": [cfg.gpu],
                "output_device": cfg.gpu,
                "find_unused_parameters": True,
            }
            model_1 = torch.nn.parallel.DistributedDataParallel(model_1, **ddp_cfg)
            model_2 = torch.nn.parallel.DistributedDataParallel(model_2, **ddp_cfg)
        elif cfg.total_gpus > 1:
            model_1 = torch.nn.DataParallel(model_1)
            model_2 = torch.nn.DataParallel(model_2)
     
        # build optimizer
        optimizer = build_optimizer([model_1, model_2], **cfg.TRAIN.OPTIM)
        #build runner
        runner = MMTRunner(
         cfg,
         [model_1,model_2,],
         optimizer,
         criterions,
         train_loader,
         writer,
        #  best_model,
        #  list_models,
         train_loader_similarities = None,
         train_sets=train_sets,
         lr_scheduler=lr_scheduler,
         reset_optim=True,
         ) 
        
        runner.resume(cfg.work_dir / "model_best.pth")
        # final testing
        for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):

            for idx in range(len(runner.model)):
                if cfg.TEST.datasets[i]=="msmt17" and i==0:
                    mAP = 0
                else:

                    print("==> Test on the no.{} model".format(idx))
                    # test_reid() on self.model[idx] will only evaluate the 'mean_net'
                    # for testing 'net', use self.model[idx].module.net
                    cmc, mAP = test_reid(
                        cfg,
                        runner.model[idx],
                        loader,
                        query,
                        gallery,
                        dataset_name=cfg.TEST.datasets[i],
                        visrankactiv= False,
                    )
                    print("map on "+l[i]+" domain : {}".format(mAP))
                if rank == 0:
                    print("in rank 0 test")
                    # writer.add_scalar("TEST_Map/"+str(idx),float(mAP),task+1)
                    writer.add_scalar("TEST_Map_"+l[i]+"/"+str(idx),float(mAP),task+1)

        runner.run()
    # load the best model
    runner.resume(cfg.work_dir / "model_best.pth")
    # final testing
    test_loaders, queries, galleries = build_test_dataloader(cfg, n_tasks, 0, False)
    # mAPs = []
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):

        for idx in range(len(runner.model)):
            if cfg.TEST.datasets[i]=="msmt17" and i==0:
                mAP = 0
            else:

                print("==> Test on the no.{} model".format(idx))
                # test_reid() on self.model[idx] will only evaluate the 'mean_net'
                # for testing 'net', use self.model[idx].module.net
                cmc, mAP = test_reid(
                    cfg,
                    runner.model[idx],
                    loader,
                    query,
                    gallery,
                    dataset_name=cfg.TEST.datasets[i],
                    visrankactiv= False,
                )
                print("map on "+l[i]+" domain : {}".format(mAP))
            if rank == 0:
                print("in rank 0 test")
                writer.add_scalar("TEST_Map_"+l[i]+"/"+str(idx),float(mAP),n_tasks)
            # mAPs.append(mAP)
            print("map: {}".format(mAP))

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
