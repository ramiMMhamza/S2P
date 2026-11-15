"""Training runners: MMT and SpCL in the online setting, with the S2P losses.

``MMTRunner.train_step`` and ``SpCLRunner.update_labels / train_step`` extend
the runners of OpenUnReID ``tools/MMT/main.py`` and ``tools/SpCL/main.py``
(Yixiao Ge, Apache-2.0) with a support-set batch and the two S2P terms.
"""

import os.path as osp
import shutil

import torch
from openunreid.apis import BaseRunner, batch_processor
from openunreid.core.metrics.accuracy import accuracy
from openunreid.data import build_train_dataloader
from openunreid.utils.dist_utils import synchronize
from torch.nn.parallel import DataParallel

from .losses import KD_INPUTS, MMDLoss, build_kd_loss
from .models import BackboneTap

__all__ = ["RUNNERS", "MMTRunner", "OUDARunner", "SpCLRunner"]


class OUDARunner(BaseRunner):
    """Runner for one task of the target stream.

    On top of ``BaseRunner`` it

    * trains for ``cfg.TRAIN.epochs`` epochs starting at ``start_epoch``
      (epochs are counted over the whole stream),
    * appends a support-set batch to every training batch (``support_loader``,
      from the second task on) and provides the S2P terms to ``train_step``,
    * always keeps the *latest* weights: in the online setting a model cannot
      be selected with labelled target data, so the validation mAP is only
      monitored,
    * logs to TensorBoard on rank 0.
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        task_id=0,
        start_epoch=0,
        support_loader=None,
        writer=None,
        meter_formats=None,
        **kwargs,
    ):
        if meter_formats is None:
            meter_formats = {"Time": ":.3f", "Acc@1": ":.2%"}
        self.kd_weight = float(cfg.S2P.kd_weight)
        self.mmd_weight = float(cfg.S2P.mmd_weight)
        self.kd_name = cfg.S2P.kd
        self.kd_input = KD_INPUTS[self.kd_name]
        self.support_loader = support_loader
        self.use_kd = support_loader is not None and self.kd_weight > 0
        self.use_mmd = self.mmd_weight > 0
        if self.use_kd:
            meter_formats["kd_" + self.kd_name] = ":.3f"
        if self.use_mmd:
            meter_formats["mmd"] = ":.3f"

        super().__init__(
            cfg,
            model,
            optimizer,
            criterions,
            train_loader,
            meter_formats=meter_formats,
            **kwargs,
        )

        models = model if isinstance(model, list) else [model]
        assert not isinstance(models[0], DataParallel), (
            "S2P relies on the [target | source | support] layout of a batch, which "
            "DataParallel breaks: use one GPU or a distributed launcher"
        )

        self.task_id = task_id
        self.writer = writer
        self._start_epoch = start_epoch
        self.kd_loss = build_kd_loss(self.kd_name).cuda() if self.use_kd else None
        self.mmd_loss = MMDLoss().cuda() if self.use_mmd else None
        self.taps = None
        if self.use_kd and self.kd_input == "activation":
            self.taps = [BackboneTap(m) for m in models]

    # ------------------------------------------------------------------ loop

    def run(self):
        end_epoch = self._start_epoch + self.cfg.TRAIN.epochs
        for ep in range(self._start_epoch, end_epoch):
            self._epoch = ep

            # generate pseudo labels
            if self.is_pseudo and (
                ep % self.cfg.TRAIN.PSEUDO_LABELS.freq == 0 or ep == self._start_epoch
            ):
                self.update_labels()
                synchronize()

            self.train()
            synchronize()
            self._log_train_meters()

            if (ep + 1) % self.cfg.TRAIN.val_freq == 0 or (ep + 1) == end_epoch:
                if "val_dataset" in self.cfg.TRAIN:
                    self._log_scalar("val/mAP", self.val())
                self.save()

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            synchronize()

        if self.taps is not None:
            for tap in self.taps:
                tap.remove()
        self._keep_task_checkpoint()

    def train(self):
        if self.support_loader is not None:
            self.support_loader.new_epoch(self._epoch)
        super().train()

    def save(self, mAP=None):
        # the latest weights, never "the best on the validation set"
        if self._rank == 0:
            self.save_model(False, self.cfg.work_dir)

    def _keep_task_checkpoint(self):
        if self._rank == 0:
            shutil.copy(
                osp.join(self.cfg.work_dir, "checkpoint.pth"),
                osp.join(self.cfg.work_dir, f"task_{self.task_id}.pth"),
            )
        synchronize()

    # --------------------------------------------------------------- logging

    def _log_scalar(self, tag, value):
        if self.writer is not None and self._rank == 0:
            self.writer.add_scalar(tag, float(value), self._epoch)

    def _log_train_meters(self):
        for name, meter in self.train_progress.meters.items():
            if name != "Time":
                self._log_scalar("train/" + name, meter.avg)

    # ------------------------------------------------------------ S2P pieces

    def _with_support(self, views):
        """Append a support-set batch to each augmented view of the batch.

        Returns the extended views and the number of regular
        (target + source) samples.
        """
        num_regular = views[0].size(0)
        if self.support_loader is None:
            return views, num_regular
        support = batch_processor(self.support_loader.next(), self.cfg.MODEL.dsbn)
        return (
            [torch.cat((view, support["img"][i].cuda())) for i, view in enumerate(views)],
            num_regular,
        )

    def _s2p_terms(self, student, teacher, num_regular, tap=None):
        """The S2P losses of one teacher-student pair.

        ``student`` / ``teacher`` are the raw outputs of the network and of its
        mean network on ``[target | source | support]``.

        * MMD aligns the student's target features with the teacher's source
          features.
        * The distillation term preserves, on the support set, the outputs of
          the teacher (mean) network.
        """
        terms = {}
        if self.use_mmd:
            half = num_regular // 2
            terms["mmd"] = self.mmd_loss(
                teacher["feat"][half:num_regular], student["feat"][:half]
            )
        if self.use_kd:
            if self.kd_input == "activation":
                student_out, teacher_out = tap.student, tap.teacher
            else:
                student_out, teacher_out = student[self.kd_input], teacher[self.kd_input]
            terms["kd_" + self.kd_name] = self.kd_loss(
                student_out[num_regular:], teacher_out[num_regular:].detach()
            )
        return terms

    def _s2p_weight(self, name):
        return self.mmd_weight if name == "mmd" else self.kd_weight


def _crop(results, num_samples, num_classes=None):
    """Keep the regular samples (and the active classes) of a model output."""
    results = {key: value[:num_samples] for key, value in results.items()}
    if num_classes is not None and "prob" in results:
        results["prob"] = results["prob"][:, :num_classes]
    return results


class MMTRunner(OUDARunner):
    """Mutual Mean-Teaching (Ge et al., ICLR 2020) + S2P."""

    def train_step(self, iter, batch):
        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        targets = data["id"].cuda()
        views, num_regular = self._with_support(
            [data["img"][0].cuda(), data["img"][1].cuda()]
        )

        results_1, results_1_mean = self.model[0](views[0])
        results_2, results_2_mean = self.model[1](views[1])

        s2p_terms = {}
        taps = self.taps or [None, None]
        for name, value in self._s2p_terms(
            results_1, results_1_mean, num_regular, taps[0]
        ).items():
            s2p_terms[name] = value
        for name, value in self._s2p_terms(
            results_2, results_2_mean, num_regular, taps[1]
        ).items():
            s2p_terms[name] = s2p_terms[name] + value

        num_classes = self.train_loader.loader.dataset.num_pids
        results_1 = _crop(results_1, num_regular, num_classes)
        results_1_mean = _crop(results_1_mean, num_regular, num_classes)
        results_2 = _crop(results_2, num_regular, num_classes)
        results_2_mean = _crop(results_2_mean, num_regular, num_classes)

        total_loss = 0
        meters = {}
        for key in self.criterions:
            if key == "soft_entropy":
                loss = self.criterions[key](
                    results_1, results_2_mean
                ) + self.criterions[key](results_2, results_1_mean)
            elif key == "soft_softmax_triplet":
                loss = self.criterions[key](
                    results_1, targets, results_2_mean
                ) + self.criterions[key](results_2, targets, results_1_mean)
            else:
                loss = self.criterions[key](results_1, targets) + self.criterions[key](
                    results_2, targets
                )
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()

        for name, loss in s2p_terms.items():
            total_loss += loss * self._s2p_weight(name)
            meters[name] = loss.item()

        acc_1 = accuracy(results_1["prob"].data, targets.data)
        acc_2 = accuracy(results_2["prob"].data, targets.data)
        meters["Acc@1"] = (acc_1[0] + acc_2[0]) * 0.5
        self.train_progress.update(meters)

        return total_loss


class SpCLRunner(OUDARunner):
    """Self-paced Contrastive Learning (Ge et al., NeurIPS 2020) + S2P."""

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
        pseudo_labels, _label_centers = self.label_generator(
            self._epoch, memory_features=memory_features, print_freq=self.print_freq
        )

        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
            self.cfg, pseudo_labels, self.train_sets, self._epoch, joint=False
        )

        # update memory labels
        memory_labels = []
        start_pid = 0
        for idx, dataset in enumerate(self.train_sets):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                labels = pseudo_labels[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
                memory_labels.append(torch.LongTensor(labels) + start_pid)
                start_pid += max(labels) + 1
            else:
                num_pids = dataset.num_pids
                memory_labels.append(torch.arange(start_pid, start_pid + num_pids))
                start_pid += num_pids
        memory_labels = torch.cat(memory_labels).view(-1)
        self.criterions["hybrid_memory"]._update_label(memory_labels)

        print(f"\n{sep} Finished updating pseudo label {sep}\n")

    def train_step(self, iter, batch):
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
        targets = data["id"].cuda()
        indexes = data["ind"].cuda()
        views, num_regular = self._with_support([data["img"][0].cuda()])

        results, results_mean = self.model(views[0])
        s2p_terms = self._s2p_terms(
            results, results_mean, num_regular, self.taps[0] if self.taps else None
        )
        results = _crop(results, num_regular)

        total_loss = 0
        meters = {}
        for key in self.criterions:
            if key == "hybrid_memory":
                loss = self.criterions[key](results, indexes)
            else:
                loss = self.criterions[key](results, targets)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()

        for name, loss in s2p_terms.items():
            total_loss += loss * self._s2p_weight(name)
            meters[name] = loss.item()

        self.train_progress.update(meters)

        return total_loss


RUNNERS = {"mmt": MMTRunner, "spcl": SpCLRunner}
