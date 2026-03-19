import torch
from roadside_radar_seg.models import build_model
from roadside_radar_seg.data import build_train_loader, build_val_loader
from roadside_radar_seg.evaluation import build_map_evaluator, build_cm_evaluator
from roadside_radar_seg.solver import build_optimizer, build_lr_scheduler
from tqdm import tqdm
from roadside_radar_seg.structures import RadarCluster3dList
from typing import List
from yacs.config import CfgNode
from torchmetrics import Metric
import numpy as np
import numpy.lib.recfunctions as rfn
from roadside_radar_seg.utils import (
    get_epoch_stats_dict,
    print_epoch_results,
    log_validation_dict_to_tensorboard,
    log_training_dict_to_tensorboard,
    plot_grad_flow,
)
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter


class DefaultPredictor:

    def __init__(self, cfg: CfgNode) -> None:

        self.cfg = cfg.clone()
        self.model = build_model(cfg)
        self.model = self.model.to(cfg.MODEL.DEVICE)
        self.model.eval()
        self.model.test()
        self.load_checkpoint()

    def __call__(self, input_tensor, radar_raw_cloud):

        with torch.inference_mode():
            return self.model(input_tensor, radar_raw_cloud)

    def load_checkpoint(self):

        checkpoint = self.cfg.MODEL.WEIGHTS

        assert Path(checkpoint).is_file()

        self.model.load_state_dict(torch.load(str(checkpoint))["model_state_dict"])


class DefaultTrainer:

    def __init__(self, cfg: CfgNode) -> None:

        self.model = build_model(cfg)

        self.optimizer = build_optimizer(cfg, self.model)

        self.train_data_loader = build_train_loader(cfg)
        self.val_data_loader = build_val_loader(cfg)
        # computes confusion matix for every epoch - (only segm head)
        self.train_cm_evaluator = build_cm_evaluator(cfg)
        self.val_cm_evaluator = build_cm_evaluator(cfg)
        # computes mAP for every epoch (SEGM + ATTN)
        self.train_map_evaluator = build_map_evaluator(cfg)
        self.val_map_evaluator = build_map_evaluator(cfg)

        self.scheduler = build_lr_scheduler(cfg, self.optimizer)

        self.start_epoch = 1
        self.max_epoch = cfg.SOLVER.MAX_EPOCHS
        self.cfg = cfg
        self.epoch_idx = 1

        self.epoch_stats_train: List[dict] = []
        self.epoch_stats_val: List[dict] = []

        self.per_sample_train_loss_dict = {}
        self.per_sample_val_loss_dict = {}

        self.writer = SummaryWriter(
            log_dir=str(Path(self.cfg.OUTPUT_DIR).joinpath("runs"))
        )

        self.sem_seg_frozen = False

    def load_previous_checkpoint(self, checkpoint: str):

        assert Path(checkpoint).is_file()

        checkpoint_dict = torch.load(str(checkpoint))

        self.start_epoch = self.epoch_idx = checkpoint_dict["epoch"]
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])

        self.epoch_stats_train = checkpoint_dict["train_stats"]
        self.epoch_stats_val = checkpoint_dict["val_stats"]

    def train_one_epoch(self):

        self.model = self.model.train()

        if self.sem_seg_frozen:
            self.model.input_embeddings.eval()
            # self.model.input_embeddings.testing = True

            self.model.backbone.eval()
            # self.model.backbone.testing = True

            self.model.segm_head.eval()
            # self.model.segm_head.testing = True


        epoch_loss_dict = {"segm_loss": [], "sim_loss": []}

        self.per_sample_train_loss_dict = {}

        with tqdm(self.train_data_loader, unit="batch") as tepoch:

            for batch_id, batch_data in enumerate(tepoch):

                self.optimizer.zero_grad()

                tepoch.set_description(f"Train Epoch {self.epoch_idx}")

                input_tensor, gt_targets, raw_input_cloud_array = batch_data

                loss_dict, radar_cluster_list, cls_labels, gt_labels, out_features = (
                    self.model(input_tensor, raw_input_cloud_array, gt_targets)
                )

                self.update_map_evaluator(
                    self.train_map_evaluator, radar_cluster_list, gt_targets
                )

                self.update_cm_evaluator(self.train_cm_evaluator, cls_labels, gt_labels)

                sim_loss = self.cfg.SOLVER.SIM_LOSS_WEIGHT * loss_dict["sim_loss"]
                segm_loss = self.cfg.SOLVER.SEGM_LOSS_WEIGHT * loss_dict["segm_loss"]

                total_loss = sim_loss + segm_loss

                if len(gt_targets) == 1:
                    self.per_sample_train_loss_dict[gt_targets[0]["name"]] = {
                        "sim_loss": loss_dict["sim_loss"].item(),
                        "segm_loss": loss_dict["segm_loss"].item(),
                        "total_loss": total_loss.item(),
                    }

                #  DEBUG
                if torch.isnan(total_loss):
                    print("total loss = nan")
                    exit(1)

                epoch_loss_dict["segm_loss"].append(segm_loss.item())
                epoch_loss_dict["sim_loss"].append(sim_loss.item())

                tepoch.set_postfix(
                    lr=f"{self.optimizer.param_groups[0]['lr']}",
                    sim=f"{sim_loss.item():.3f}",
                    cls=f"{segm_loss.item():.3f}",
                    total=f"{total_loss.item():.3f}",
                )
                try:
                    if self.sem_seg_frozen:
                        sim_loss.backward()
                    else:
                        total_loss.backward()
                except:
                    print(total_loss.requires_grad, sim_loss.requires_grad, segm_loss.requires_grad)
                # gradient clipping
                # must be called after backward.
                if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.SOLVER.CLIP_GRADIENTS_NORM_VALUE,
                    )

                self.optimizer.step()

        return epoch_loss_dict

    def validate_one_epoch(self):

        self.model = self.model.eval()

        epoch_loss_dict = {"segm_loss": [], "sim_loss": []}

        self.per_sample_val_loss_dict = {}

        with torch.no_grad():

            with tqdm(self.val_data_loader, unit="batch") as tepoch:

                for batch_id, batch_data in enumerate(tepoch):

                    tepoch.set_description(f"Val Epoch {self.epoch_idx}")

                    input_tensor, gt_targets, raw_input_cloud_array = batch_data

                    (
                        loss_dict,
                        radar_cluster_list,
                        cls_labels,
                        gt_labels,
                        out_features,
                    ) = self.model(input_tensor, raw_input_cloud_array, gt_targets)
                    self.update_map_evaluator(
                        self.val_map_evaluator, radar_cluster_list, gt_targets
                    )

                    self.update_cm_evaluator(
                        self.val_cm_evaluator, cls_labels, gt_labels
                    )

                    sim_loss = self.cfg.SOLVER.SIM_LOSS_WEIGHT * loss_dict["sim_loss"]

                    segm_loss = (
                        self.cfg.SOLVER.SEGM_LOSS_WEIGHT * loss_dict["segm_loss"]
                    )

                    total_loss = sim_loss + segm_loss

                    if len(gt_targets) == 1:
                        self.per_sample_val_loss_dict[gt_targets[0]["name"]] = {
                            "sim_loss": sim_loss.item(),
                            "segm_loss": segm_loss.item(),
                            "total_loss": total_loss.item(),
                        }

                    #  DEBUG
                    if torch.isnan(total_loss):
                        print("total loss = nan")
                        exit(1)

                    epoch_loss_dict["segm_loss"].append(segm_loss.item())
                    epoch_loss_dict["sim_loss"].append(sim_loss.item())

                    tepoch.set_postfix(
                        sim=f"{(sim_loss.item() if isinstance(sim_loss, torch.Tensor) else sim_loss):.3f}",
                        cls=f"{segm_loss.item():.3f}",
                        total=f"{total_loss.item():.3f}",
                    )
        return epoch_loss_dict

    def update_cm_evaluator(
        self, cm_evaluator: Metric, pred_labels: torch.Tensor, gt_labels: torch.Tensor
    ):
        if pred_labels.ndim == 3:
            pred_labels = pred_labels.squeeze(-1)

        gt_labels = gt_labels.to(pred_labels)

        valid_points_mask = pred_labels != -1

        # [N] -> sum(N1 + N2 + ...... Nbatchsize)
        gt_labels = gt_labels[valid_points_mask]
        pred_labels = pred_labels[valid_points_mask]

        cm_evaluator.update(preds=pred_labels, target=gt_labels)

    def update_map_evaluator(
        self,
        evaluator: Metric,
        radar_cluster_list: List[RadarCluster3dList],
        gt_target_list: List[dict],
    ):

        device = self.cfg.MODEL.DEVICE

        assert len(radar_cluster_list) == len(
            gt_target_list
        ), f"number of samples must be same across gt and the network predictions."

        gt_list_dict_for_map = []
        for gt in gt_target_list:
            current_gt_dict = {"clusters": [], "labels": gt["labels"].to(device)}
            for i in gt["points"]:
                gt_cluster = i[["x", "y", "z"]]
                gt_cluster_tensor = torch.from_numpy(
                    rfn.structured_to_unstructured(gt_cluster).copy()
                )
                current_gt_dict["clusters"].append(gt_cluster_tensor.to(device))
            gt_list_dict_for_map.append(current_gt_dict)

        # convert List[RadarCluster3Dlist] to format acceptable by the evaluator
        pred_clusters = []
        for cl_list in radar_cluster_list:
            # RadarCluster3dList for one sample
            current_res_dict = {"clusters": [], "labels": [], "scores": []}

            if len(cl_list) == 0:
                pred_clusters.append(
                    {
                        "clusters": [],  # torch.Tensor().to(self.cfg.MODEL.DEVICE),
                        "labels": torch.Tensor().to(self.cfg.MODEL.DEVICE),
                        "scores": torch.Tensor().to(self.cfg.MODEL.DEVICE),
                    }
                )
                continue

            for cluster in cl_list:
                cluster_cloud = cluster.radar_points  # recarray
                cluster_label = cluster.category.value  # int
                cluster_score = cluster.category_confidence  # float

                cluster_xyz = cluster_cloud[["x", "y", "z"]]
                cluster_tensor = torch.from_numpy(
                    rfn.structured_to_unstructured(cluster_xyz).copy()
                )

                current_res_dict["clusters"].append(cluster_tensor.to(device))
                current_res_dict["labels"].append(cluster_label)
                current_res_dict["scores"].append(cluster_score)

            # sanity check
            assert (
                len(current_res_dict["clusters"])
                == len(current_res_dict["labels"])
                == len(current_res_dict["scores"])
            )

            current_res_dict["labels"] = torch.tensor(current_res_dict["labels"]).to(
                device
            )
            current_res_dict["scores"] = torch.tensor(current_res_dict["scores"]).to(
                device
            )

            pred_clusters.append(current_res_dict)

        assert len(pred_clusters) == len(gt_list_dict_for_map)

        evaluator.update(preds=pred_clusters, target=gt_list_dict_for_map)

    def save_checkpoint(self):

        output_dir = self.cfg.OUTPUT_DIR
        torch.save(
            {
                "epoch": self.epoch_idx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_stats": self.epoch_stats_train,
                "val_stats": self.epoch_stats_val,
            },
            Path(output_dir).joinpath(
                f"model_epoch_{str(self.epoch_idx).zfill(3)}.pth"
            ),
        )

    def save_progress_jsons(self):

        current_epoch_idx = self.epoch_idx

        train_save_name = f"results_train_epoch_{str(current_epoch_idx).zfill(3)}.json"
        train_to_dump = {str(i["epoch"]).zfill(3): i for i in self.epoch_stats_train}

        with open(str(Path(self.cfg.OUTPUT_DIR).joinpath(train_save_name)), "w") as f:
            json.dump(train_to_dump, f, indent=4)

        val_save_name = f"results_val_epoch_{str(current_epoch_idx).zfill(3)}.json"
        val_to_dump = {str(i["epoch"]).zfill(3): i for i in self.epoch_stats_val}

        with open(str(Path(self.cfg.OUTPUT_DIR).joinpath(val_save_name)), "w") as f:
            json.dump(val_to_dump, f, indent=4)

        # per frame loss values for current epoch
        train_per_sample_loss_save_name = (
            f"per_frame_losses_train_epoch_{str(current_epoch_idx).zfill(3)}.json"
        )

        with open(
            str(Path(self.cfg.OUTPUT_DIR).joinpath(train_per_sample_loss_save_name)),
            "w",
        ) as f:
            json.dump(self.per_sample_train_loss_dict, f, indent=4)

        val_per_sample_loss_save_name = (
            f"per_frame_losses_val_epoch_{str(current_epoch_idx).zfill(3)}.json"
        )

        with open(
            str(Path(self.cfg.OUTPUT_DIR).joinpath(val_per_sample_loss_save_name)), "w"
        ) as f:
            json.dump(self.per_sample_val_loss_dict, f, indent=4)

    def freeze_semseg_weights(self):

        for name, param in self.model.input_embeddings.named_parameters():
            param.requires_grad = False

        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = False

        for name, param in self.model.segm_head.named_parameters():
            param.requires_grad = False


        self.sem_seg_frozen = True


    def load_checkpoint(self):
        assert Path(self.cfg.MODEL.WEIGHTS).is_file()

        checkpoint_dict = torch.load(str(self.cfg.MODEL.WEIGHTS))
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])

    def train(self):

        checkpoint_period = self.cfg.SOLVER.CHECKPOINT_PERIOD

        for epoch_idx in range(self.start_epoch, self.max_epoch):

            self.epoch_idx = epoch_idx

            train_epoch_loss_dict = self.train_one_epoch()

            # plot_grad_flow(self.model.named_parameters())

            with torch.no_grad():
                val_epoch_loss_dict = self.validate_one_epoch()

            train_evaluation_dict = self.train_map_evaluator.compute()
            self.train_map_evaluator.reset()

            val_evaluation_dict = self.val_map_evaluator.compute()
            self.val_map_evaluator.reset()

            train_cm = self.train_cm_evaluator.compute()
            self.train_cm_evaluator.reset()

            val_cm = self.val_cm_evaluator.compute()
            self.val_cm_evaluator.reset()
            print(train_evaluation_dict)

            train_epoch_stats_dict = get_epoch_stats_dict(
                sim_losses_batchwise=train_epoch_loss_dict["sim_loss"],
                cls_losses_batchwise=train_epoch_loss_dict["segm_loss"],
                map_dict=train_evaluation_dict,
                lerning_rate=self.optimizer.param_groups[0]["lr"],
                epoch_idx=self.epoch_idx,
                cm=train_cm,
            )

            val_epoch_stats_dict = get_epoch_stats_dict(
                sim_losses_batchwise=val_epoch_loss_dict["sim_loss"],
                cls_losses_batchwise=val_epoch_loss_dict["segm_loss"],
                map_dict=val_evaluation_dict,
                lerning_rate=self.optimizer.param_groups[0]["lr"],
                epoch_idx=self.epoch_idx,
                cm=val_cm,
            )

            self.epoch_stats_train.append(train_epoch_stats_dict)
            self.epoch_stats_val.append(val_epoch_stats_dict)

            if self.cfg.SOLVER.LR_SCHEDULER_NAME == "ReduceLROnPlateau":
                self.scheduler.step(val_epoch_stats_dict["total_loss"])
            else:
                self.scheduler.step()

            self.save_progress_jsons()

            log_training_dict_to_tensorboard(train_epoch_stats_dict, writer=self.writer)
            log_validation_dict_to_tensorboard(val_epoch_stats_dict, writer=self.writer)

            print_epoch_results(
                train_epoch_dict=train_epoch_stats_dict,
                val_epoch_dict=val_epoch_stats_dict,
            )

            if (self.epoch_idx) % checkpoint_period == 0:
                self.save_checkpoint()
