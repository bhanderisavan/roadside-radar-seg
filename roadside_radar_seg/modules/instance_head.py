import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Union
from roadside_radar_seg.configs import configurable
from roadside_radar_seg.modules import build_self_attention, MLP
from torchvision.ops.focal_loss import sigmoid_focal_loss
from roadside_radar_seg.structures import (
    RadarCluster3dList,
    RadarCluster3d,
    TimeStamp,
    ClusterCentroidTuple,
    ObjectCategory,
)
import datetime as dt
import numpy as np
import time
from roadside_radar_seg.utils import index_recarray_by_column
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class InstanceHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        nn_features_list: List = [],
        activations: List = [],  # len = len(nn_features_list) - 1
        normalizations: List = [],  # len = len(nn_features_list) - 1
        layer_type: List = [],  # conv or linear,
        loss_name: str,
        attention_head: nn.Module,
        similarity_threshold: float,
        add_gt_points: bool,
        add_gt_points_thresh: float = 0.0,
        input_conf_thresh_train: float = 0.0,
        input_conf_thresh_test: float = 0.0,
    ) -> None:

        super().__init__()

        self.attention = attention_head

        self.mlp = []
        if nn_features_list:
            self.mlp = nn.Sequential(
                *[
                    nn.Sequential(
                        MLP(
                            in_features=nn_features_list[i - 1],
                            out_features=nn_features_list[i],
                            activation=activations[i - 1],
                            normalization=normalizations[i - 1],
                            use_conv1d=layer_type[i - 1] == "conv",
                        )
                    )
                    for i in range(1, len(nn_features_list))
                ]
            )

        self.loss_name = loss_name

        self.similarity_threshold = similarity_threshold
        self.add_gt_points = add_gt_points
        self.add_gt_points_thresh = add_gt_points_thresh

        self.input_conf_thresh_train = input_conf_thresh_train
        self.input_conf_thresh_test = input_conf_thresh_test

    @classmethod
    def from_config(cls, cfg):
        self_attention = build_self_attention(cfg)

        return {
            "attention_head": self_attention,
            "nn_features_list": cfg.MODEL.INSTANCE_HEAD.NN_LAYERS_LIST,
            "activations": cfg.MODEL.INSTANCE_HEAD.ACTIVATIONS_LIST,
            "normalizations": cfg.MODEL.INSTANCE_HEAD.NORMALIZATIONS_LIST,
            "layer_type": cfg.MODEL.INSTANCE_HEAD.LAYER_TYPES_LIST,
            "loss_name": cfg.MODEL.INSTANCE_HEAD.LOSS,
            "similarity_threshold": cfg.MODEL.INSTANCE_HEAD.ATTENTION.SIMILARTY_THRESHOLD,
            "add_gt_points": cfg.MODEL.INSTANCE_HEAD.ADD_GT_POINTS,
            "add_gt_points_thresh": cfg.MODEL.INSTANCE_HEAD.ADD_GT_POINTS_THRESHOLD,
            "input_conf_thresh_train": cfg.MODEL.INSTANCE_HEAD.INPUT_CONFIDENCE_THRESH_TRAIN,
            "input_conf_thresh_test": cfg.MODEL.INSTANCE_HEAD.INPUT_CONFIDENCE_THRESH_TEST,
        }

    def generate_gt_similarity(self, input_point_indices, gt_dict):

        gt_similarity = torch.eye(len(input_point_indices))

        list_of_instances = [obj["index"].tolist() for obj in gt_dict["points"]]

        for ip_idx, ip_point_ind in enumerate(input_point_indices):

            # gt instance to which the current predicted point belongs to.
            src_point_instance = [
                i for i in list_of_instances if ip_point_ind.item() in i
            ]
            # if the predicted point does not belong to any gt instance
            if not len(src_point_instance) > 0:
                continue

            # similarity_point_ind and indices_with_same_instance both are the radar point index fields.
            indices_with_same_instance = [
                j
                for j, i in enumerate(input_point_indices)
                if i.item() in src_point_instance[0]
            ]
            # counter + ip_idx row and all of the indices_with_same_instance must have value 1.
            gt_similarity[ip_idx][indices_with_same_instance] = 1.0

        return gt_similarity

    def calculate_loss(self, predicted_similarity, gt_similarity):

        if predicted_similarity.nelement() == 0 and gt_similarity.nelement() == 0:
            return torch.Tensor([0]).to(predicted_similarity)

        # taking the average of upper and lower triangle. this will ensure symmetry.
        predicted_similarity = (predicted_similarity + predicted_similarity.T) / 2

        if self.loss_name == "binary_cross_entropy":
            gt_similarity = gt_similarity.to(predicted_similarity)

            loss = F.binary_cross_entropy_with_logits(
                predicted_similarity,
                gt_similarity,
                reduction="mean",
            )

        elif self.loss_name == "focal_loss":

            # compute alpha adaptivly based on current class distribution.
            total_samples = gt_similarity.shape[0]
            fg_samples = torch.count_nonzero(gt_similarity)
            bg_samples = total_samples - fg_samples
            alpha = torch.min(fg_samples, bg_samples) / total_samples

            loss = sigmoid_focal_loss(
                predicted_similarity, gt_similarity, reduction="mean", alpha=alpha
            )
        else:
            raise NotImplementedError()

        return loss

    def generate_results(
        self,
        pred_similarity_logits,
        input_point_indices,
        radar_raw_point_cloud,
        threshold,
        category_id,
        cls_score,
    ) -> List[RadarCluster3d]:

        assert input_point_indices.shape[0] == pred_similarity_logits.shape[0]

        return_list = []

        pred_similarity = torch.sigmoid(pred_similarity_logits)
        # taking the average of upper and lower triangle. this will ensure symmetry.
        # lower triangle  = 0
        pred_similarity = (
            torch.triu(pred_similarity) + torch.tril(pred_similarity).T
        ) / 2
        # lower traingle = upper traingle
        # pred_similarity = (pred_similarity + pred_similarity.T) / 2

        pred_similarity[pred_similarity <= threshold] = 0
        pred_similarity[pred_similarity > threshold] = 1

        # force diagonal to be 1
        pred_similarity = pred_similarity.fill_diagonal_(1)

        taken_pts: List[int] = []

        for row_idx, row in enumerate(pred_similarity):
            if row_idx in taken_pts:
                continue
            row_grp = torch.where(row == 1)[0]
            if row_grp.nelement() == 0:
                # current point similarity with itself is also < thresh.
                # ENTIRE ROW IS 0. - bg point misclassified - ignore.
                taken_pts.append(row_idx)
                continue

            # assert (
            #     row_idx in row_grp
            # ), f"self similarity 0 but other similarity 1"

            taken_pts.extend(row_grp.tolist())
            if row_idx not in row_grp:
                print("-" * 100)
                continue

            radar_points = index_recarray_by_column(
                radar_raw_point_cloud, "index", input_point_indices[row_grp].tolist()
            )

            if radar_points.size == 0:
                print("------DEBUG BE - SOMETHING BAD HAPPENED-------")
                continue

            centroid = ClusterCentroidTuple(
                x=radar_points["x"].mean(),
                y=radar_points["y"].mean(),
                z=radar_points["z"].mean(),
            )

            radar_cluster = RadarCluster3d(
                radar_points_list=radar_points,
                centroid=centroid,
                velocity=radar_points["range_rate"].mean(),
                category=ObjectCategory(int(category_id)),
                category_confidence=cls_score.item(),
            )

            return_list.append(radar_cluster)

        return return_list

    # FIXME - attention similarity is not being trained when batch size > 1.
    def forward(
        self,
        padded_logits: torch.Tensor,  # [B, N_max, N_cls]
        batched_input: Union[PackedSequence, torch.Tensor],  # [B, N_max, N_features]
        input_point_indices: torch.Tensor,
        radar_raw_point_cloud: np.recarray,
        # model_input,
        batched_tagets: List[dict] = None,
    ) -> Tuple[torch.Tensor, List[RadarCluster3dList]]:

        bs = input_point_indices.shape[0]

        # class probabilities
        cls_predictions = F.softmax(padded_logits, dim=2)  # [B, N_max, N_cls]

        # per point cls score, and cls labels
        cls_scores, cls_labels = torch.topk(
            cls_predictions, 1
        )  # [B, N_max, 1], [B, N_max, 1]

        cls_scores = cls_scores.detach()

        if not self.testing:
            if self.add_gt_points:
                for idx in range(bs):
                    current_gt = batched_tagets[idx]
                    current_input_indices = input_point_indices[idx]
                    for instnce, cat in zip(current_gt["points"], current_gt["labels"]):
                        gt_indices = torch.Tensor(instnce["index"].astype(np.float32))
                        gt_indices_in_input = torch.Tensor(
                            [
                                torch.nonzero(current_input_indices == b).item()
                                for b in gt_indices
                            ]
                        )
                        gt_indices_in_input = gt_indices_in_input.to(cls_scores)
                        replace_mask, _ = torch.where(
                            cls_scores[idx][gt_indices_in_input.long()]
                            < self.add_gt_points_thresh
                        )

                        cls_labels[idx][gt_indices_in_input.long()[replace_mask]] = cat

                        # otherwise these points will be ignored in the filtering step.
                        # cls_scores[idx][gt_indices_in_input.long()[replace_mask]] = 1.0

        if self.testing:
            low_quality_mask = cls_scores < self.input_conf_thresh_test
        else:
            low_quality_mask = cls_scores < self.input_conf_thresh_train

        # setting low quality points score to nan - will be ignored in the next step
        cls_scores[low_quality_mask] = torch.nan

        # batch size must be 1 when features are no an instance of packed sequence.
        # this is only used  during inference
        if isinstance(batched_input, torch.Tensor):
            assert batched_input.shape[0] == 1

        if self.mlp:
            if isinstance(batched_input, PackedSequence):  # and bs > 1:
                ip, batch_sizes, sorted_indices, unsorted_indices = batched_input
                ip = self.mlp(ip)

                batched_input = PackedSequence(
                    ip,
                    batch_sizes,
                    sorted_indices=sorted_indices,
                    unsorted_indices=unsorted_indices,
                )

                padding_value = input_point_indices[-1][-1].item()
                batched_input, _ = pad_packed_sequence(
                    batched_input, batch_first=True, padding_value=padding_value
                )
            else:
                batched_input = self.mlp(batched_input)

        if not self.testing:
            total_sim_loss = torch.tensor([0.0], requires_grad = True).to(cls_predictions.device)

        batch_radar_cluster_list = []

        for idx in range(bs):

            radar_frame_epoch_time = int(time.time() * 1000)
            filename = ""
            if not self.testing:
                gt_dict = batched_tagets[idx]

                filename = gt_dict["name"]
                # if filename == "radar_01__2023-07-06-17-36-53-368_bg01.pcd":
                #     print("---")
                radar_utc_time = dt.datetime.strptime(
                    gt_dict["date_captured"], "%Y-%m-%d-%H-%M-%S-%f"
                )
                radar_frame_epoch_time = (
                    radar_utc_time - dt.datetime(1970, 1, 1)
                ).total_seconds()
                radar_frame_epoch_time = int(
                    radar_frame_epoch_time * 1000
                )  # convert into milliseconds and int
                # prepare gt for this sample [rows,rows]

            current_sample = batched_input[idx]
            current_scores = cls_scores[idx]
            current_labels = cls_labels[idx]

            current_input_radar_point_cloud = radar_raw_point_cloud[idx]
            current_input_point_indices_all = input_point_indices[idx]

            # [N_max]
            # low quality points and padded points have torch.nan as their scores.
            mask = torch.logical_not(current_scores.isnan()).squeeze(1)

            # valid points
            sample = current_sample[mask]
            scores = current_scores[mask].clone()
            labels = current_labels[mask].clone()
            current_input_point_indices = current_input_point_indices_all[mask]

            unique_labels = labels.unique()

            if not self.testing:
                one_sample_sim_loss = torch.tensor([0.0]).to(cls_predictions)

            one_sample_radar_cluster_list = []
            total_to_attn = 0
            for u_label in unique_labels:

                # ignore bg class
                if u_label.item() == 0:
                    continue

                try:
                    rows, cols = torch.where(labels == u_label)
                except ValueError:
                    continue

                if rows.nelement() == 1:
                    
                    # no need to compute similarity. it is single point.
                    radar_points = index_recarray_by_column(
                        current_input_radar_point_cloud,
                        "index",
                        current_input_point_indices[rows].tolist(),
                    )

                    centroid = ClusterCentroidTuple(
                        x=radar_points["x"].mean(),
                        y=radar_points["y"].mean(),
                        z=radar_points["z"].mean(),
                    )

                    radar_cluster = RadarCluster3d(
                        radar_points_list=radar_points,
                        centroid=centroid,
                        velocity=radar_points["range_rate"].mean(),
                        category=ObjectCategory(int(u_label.item())),
                        category_confidence=scores[rows].item(),
                    )

                    one_sample_radar_cluster_list.append(radar_cluster)

                    continue

                to_attn = sample[rows]
                # logits
                _, pred_sim_logits = self.attention(to_attn)  # [rows,rows]
                total_to_attn += 1

                if not self.testing:
                    # prepare gt for this sample [rows,rows]
                    gt_sim = self.generate_gt_similarity(
                        current_input_point_indices[rows], gt_dict
                    )

                    # loss
                    one_sample_sim_loss = one_sample_sim_loss + self.calculate_loss(
                        pred_sim_logits, gt_sim
                    )

                radar_cluster_3d_list = self.generate_results(
                    pred_sim_logits.detach(),
                    current_input_point_indices[rows],
                    current_input_radar_point_cloud,
                    threshold=self.similarity_threshold,
                    category_id=u_label.item(),
                    cls_score=scores[rows].mean(),
                )

                one_sample_radar_cluster_list.extend(radar_cluster_3d_list)

            if not self.testing:
                # divide by all the non zero labels in the current frame
                # nonzero because we do not calculate similarity for the

                if total_to_attn != 0:
                    total_sim_loss = total_sim_loss + (
                        one_sample_sim_loss / total_to_attn
                    )

                else:
                    assert one_sample_sim_loss == 0

            batch_radar_cluster_list.append(
                RadarCluster3dList(
                    time_stamp=TimeStamp(radar_frame_epoch_time),
                    frame_id=idx,
                    frame_name=filename,
                    radar_clusters_3d=one_sample_radar_cluster_list,
                )
            )

        if self.testing:
            return batch_radar_cluster_list

        batch_avg_sim_loss = total_sim_loss / bs

        assert isinstance(batch_avg_sim_loss, torch.Tensor)

        return batch_avg_sim_loss, batch_radar_cluster_list


def build_instance_head(cfg):
    return InstanceHead(cfg)
