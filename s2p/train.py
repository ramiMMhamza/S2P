"""Train a re-ID model with S2P in the online UDA setting.

    python -m s2p.train configs/mmt_msmt17_to_market1501.yaml --work-dir my_run

The target stream is split into ``OUDA.n_tasks`` identity-disjoint tasks. The
base UDA method (MMT or SpCL) adapts the model to one task after the other;
from the second task on, a support set of source images similar to the target
data seen so far is used to preserve feature similarities (see ``s2p.support``).
"""

import argparse
import collections
import json
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from openunreid.apis import set_random_seed, test_reid
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader
from openunreid.models.losses import build_loss
from openunreid.models.utils.dsbn_utils import switch_target_bn
from openunreid.models.utils.extract import extract_features
from openunreid.utils.config import log_config_to_file
from openunreid.utils.dist_utils import get_dist_info, init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.torch_utils import copy_state_dict, load_checkpoint
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .config import METHODS, load_config
from .data import OUDAStream, make_eval_loaders, make_train_loader
from .dsbn import enable_support_branch
from .label_generator import TaskLabelGenerator
from .models import build_model, unwrap
from .runners import RUNNERS
from .support import select_support_set
from .viz import log_support_pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="S2P training in the online UDA setting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="yaml config file, see configs/")
    parser.add_argument("--work-dir", help="run name: outputs go to LOGS_ROOT/<work-dir> (default: config name)")
    parser.add_argument("--method", choices=METHODS, help="base UDA method (overrides METHOD)")
    parser.add_argument("--data-root", help="overrides DATA_ROOT")
    parser.add_argument("--logs-root", help="overrides LOGS_ROOT")
    parser.add_argument("--source-pretrained", help="overrides MODEL.source_pretrained")
    parser.add_argument("--backbone-path", help="overrides MODEL.backbone_path")
    parser.add_argument("--n-tasks", type=int, help="overrides OUDA.n_tasks")
    parser.add_argument("--epochs", type=int, help="epochs per task (overrides TRAIN.epochs)")
    parser.add_argument("--iters", type=int, help="iterations per epoch (overrides TRAIN.iters)")
    parser.add_argument("--kd", choices=["s2p", "lwf", "sp", "at"], help="overrides S2P.kd")
    parser.add_argument("--kd-weight", type=float, help="overrides S2P.kd_weight (0 disables)")
    parser.add_argument("--mmd-weight", type=float, help="overrides S2P.mmd_weight (0 disables)")
    parser.add_argument("--seed", type=int, help="overrides TRAIN.seed")
    parser.add_argument("--resume-from", help="task_<k>.pth checkpoint: training restarts at task k+1")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none", help="job launcher")
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set any other config key: --set KEY VALUE [KEY VALUE ...]",
    )
    return parser.parse_args(argv)


def setup(args):
    cfg = load_config(
        args.config,
        overrides={
            "METHOD": args.method,
            "DATA_ROOT": args.data_root,
            "LOGS_ROOT": args.logs_root,
            "MODEL.source_pretrained": args.source_pretrained,
            "MODEL.backbone_path": args.backbone_path,
            "OUDA.n_tasks": args.n_tasks,
            "TRAIN.epochs": args.epochs,
            "TRAIN.iters": args.iters,
            "S2P.kd": args.kd,
            "S2P.kd_weight": args.kd_weight,
            "S2P.mmd_weight": args.mmd_weight,
            "TRAIN.seed": args.seed,
        },
        set_cfgs=args.set_cfgs,
    )
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg.work_dir = cfg.LOGS_ROOT / (args.work_dir or Path(args.config).stem)
    mkdir_if_missing(cfg.work_dir)
    return cfg


class Tee:
    """Mirror stdout to a log file; only the first process (rank 0) writes."""

    def __init__(self, path, enabled):
        self.console = sys.stdout
        self.enabled = enabled
        self.file = open(path, "a") if enabled else None

    def write(self, msg):
        if self.enabled:
            self.console.write(msg)
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class Method:
    """What differs between the base UDA methods: models, losses, runner."""

    def __init__(self, cfg, stream, distributed):
        self.cfg = cfg
        self.stream = stream
        self.distributed = distributed

    def _parallel(self, model):
        model.cuda()
        if self.distributed:
            return DistributedDataParallel(
                model,
                device_ids=[self.cfg.gpu],
                output_device=self.cfg.gpu,
                find_unused_parameters=True,
            )
        if self.cfg.total_gpus > 1:
            return DataParallel(model)
        return model

    def load_weights(self, path):
        state = load_checkpoint(path)
        for idx, model in enumerate(self.models):
            key = f"state_dict_{idx + 1}" if len(self.models) > 1 else "state_dict"
            copy_state_dict(state[key], unwrap(model), strip="module.")
        return state["epoch"]

    def runner_model(self):
        return self.models if len(self.models) > 1 else self.models[0]

    def criterions(self, task_id, task_sets):
        raise NotImplementedError

    reset_optim = True
    meter_formats = None


class MMT(Method):
    def __init__(self, cfg, stream, distributed):
        super().__init__(cfg, stream, distributed)
        # The classifier must have at least as many outputs as there can be
        # clusters, which is bounded by the number of images of a task.
        self.num_classes = (
            max(stream.task_size(t) for t in range(stream.n_tasks)) + stream.source_num_pids
        )
        self.models = [
            self._parallel(
                build_model(cfg, self.num_classes, init=cfg.MODEL.source_pretrained)
            )
            for _ in range(2)
        ]
        self._criterions = build_loss(
            cfg.TRAIN.LOSS, num_classes=self.num_classes, cuda=True
        )

    def criterions(self, task_id, task_sets):
        return self._criterions


class SpCL(Method):
    reset_optim = False
    meter_formats = {"Time": ":.3f"}

    def __init__(self, cfg, stream, distributed):
        super().__init__(cfg, stream, distributed)
        self.models = [
            self._parallel(build_model(cfg, 0, init=cfg.MODEL.source_pretrained))
        ]

    def criterions(self, task_id, task_sets):
        """A hybrid memory sized and initialised for the current task."""
        cfg, model = self.cfg, self.models[0]
        unsup = cfg.TRAIN.unsup_dataset_indexes
        num_memory = sum(
            len(s) if idx in unsup else s.num_pids for idx, s in enumerate(task_sets)
        )
        criterions = build_loss(
            cfg.TRAIN.LOSS,
            num_features=unwrap(model).net.num_features,
            num_memory=num_memory,
            cuda=True,
        )

        # instance features for the unlabelled target, class centres for the source
        eval_sets = [
            self.stream.target(task_id, train=False, keep_labels=True),
            self.stream.source(train=False),
        ]
        memory_features = []
        for idx, (loader, dataset) in enumerate(
            zip(make_eval_loaders(cfg, eval_sets), eval_sets)
        ):
            features = extract_features(
                model, loader, dataset.data, with_path=False, prefix="Extract: "
            )
            assert features.size(0) == len(dataset)
            if idx in unsup:
                memory_features.append(features)
            else:
                centers = collections.defaultdict(list)
                for feature, (_, pid, _) in zip(features, dataset.data):
                    centers[pid].append(feature.unsqueeze(0))
                memory_features.append(
                    torch.stack(
                        [torch.cat(centers[pid], 0).mean(0) for pid in sorted(centers)], 0
                    )
                )
        criterions["hybrid_memory"]._update_feature(torch.cat(memory_features))
        return criterions


_METHODS = {"mmt": MMT, "spcl": SpCL}


def evaluate(cfg, method, test_data, task_id, writer, results):
    """Test every model on every dataset of ``cfg.TEST.datasets``."""
    target_bn_idx = 0  # training datasets are ordered [target, source]
    rank, _, _ = get_dist_info()
    for loader, query, gallery, name in zip(*test_data, cfg.TEST.datasets):
        for idx, model in enumerate(method.models):
            cmc, mAP = test_reid(cfg, model, loader, query, gallery, dataset_name=name)
            if rank == 0:
                results.append(
                    {
                        "after_task": task_id,
                        "dataset": name,
                        "model": idx,
                        "mAP": float(mAP),
                        "rank1": float(cmc[0]),
                    }
                )
                if writer is not None:
                    writer.add_scalar(f"test_mAP/{name}/model_{idx}", float(mAP), task_id + 1)
                    writer.add_scalar(f"test_rank1/{name}/model_{idx}", float(cmc[0]), task_id + 1)
            # test_reid() leaves the model on the BN of the tested domain;
            # clustering and validation need the target one
            if cfg.MODEL.dsbn:
                switch_target_bn(model, target_bn_idx)
    if rank == 0:
        with open(cfg.work_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)


def main(argv=None):
    args = parse_args(argv)
    cfg = setup(args)

    distributed = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    rank, _, _ = get_dist_info()
    synchronize()

    logger = Tee(cfg.work_dir / "log.txt", enabled=rank == 0)
    sys.stdout = logger
    try:
        return run(args, cfg, distributed)
    finally:
        sys.stdout = logger.console
        logger.close()


def run(args, cfg, distributed):
    start_time = time.monotonic()
    rank, _, _ = get_dist_info()
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        shutil.copy(args.config, cfg.work_dir / "config.yaml")
        writer = SummaryWriter(str(cfg.work_dir / "tb"), flush_secs=10)
    print(f"==========\nArgs:{args}\n==========")
    log_config_to_file(cfg)

    n_tasks = cfg.OUDA.n_tasks
    stream = OUDAStream(cfg, n_tasks)
    print(stream.summary())
    method = _METHODS[cfg.METHOD](cfg, stream, distributed)
    test_data = build_test_dataloader(cfg)
    results = []

    first_task, start_epoch = 0, 0
    if args.resume_from:
        start_epoch = method.load_weights(args.resume_from)
        first_task, remainder = divmod(start_epoch, cfg.TRAIN.epochs)
        if remainder != 0 or first_task >= n_tasks:
            raise ValueError(
                "--resume-from expects the checkpoint saved at the end of a task "
                f"(task_<k>.pth, k < {n_tasks - 1}) of a run with the same TRAIN.epochs"
            )
        print(f"Resumed from {args.resume_from}: restarting at task {first_task}")
    else:
        # source-only reference, before any adaptation
        evaluate(cfg, method, test_data, -1, writer, results)

    source_set = stream.source(train=True)
    for task_id in range(first_task, n_tasks):
        print(f"\n########## Task {task_id + 1} / {n_tasks} ##########")

        support_loader = None
        if task_id > 0:
            # support set: source identities similar to the previous task
            previous_target = stream.target(task_id - 1, train=True, keep_labels=True)
            support_set, selection = select_support_set(
                cfg, method.models[0], previous_target, source_set
            )
            print(
                f"Support set: {len(selection.pids)} source identities, {len(support_set)} images (from {selection.num_target} target x {selection.num_source} "
                "source sampled images)"
            )
            if cfg.S2P.log_images and rank == 0:
                log_support_pairs(writer, task_id, previous_target, source_set, selection)
            support_loader = make_train_loader(cfg, [support_set], joint=True)
            for model in method.models:
                enable_support_branch(model)

        train_sets = [stream.target(task_id, train=True), source_set]
        cluster_sets = [stream.target(task_id, train=False, keep_labels=True)]
        optimizer = build_optimizer(method.models, **cfg.TRAIN.OPTIM)
        lr_scheduler = None
        if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
            lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)

        runner = RUNNERS[cfg.METHOD](
            cfg,
            method.runner_model(),
            optimizer,
            method.criterions(task_id, train_sets),
            make_train_loader(cfg, train_sets, joint=False),
            task_id=task_id,
            start_epoch=task_id * cfg.TRAIN.epochs,
            support_loader=support_loader,
            writer=writer,
            train_sets=train_sets,
            lr_scheduler=lr_scheduler,
            meter_formats=dict(method.meter_formats) if method.meter_formats else None,
            reset_optim=method.reset_optim,
            label_generator=TaskLabelGenerator(
                cfg,
                method.models,
                make_eval_loaders(cfg, cluster_sets),
                cluster_sets,
            ),
        )
        runner.run()
        evaluate(cfg, method, test_data, task_id, writer, results)

    if writer is not None:
        writer.close()
    print("Total running time: ", timedelta(seconds=time.monotonic() - start_time))
    return results


if __name__ == "__main__":
    main()
