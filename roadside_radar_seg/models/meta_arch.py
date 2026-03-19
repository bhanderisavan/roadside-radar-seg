from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from roadside_radar_seg.configs import configurable
from roadside_radar_seg.modules import (
    build_backbone,
    build_feature_normalizer,
    build_input_embeddings,
    build_instance_head,
    build_segm_head,
)
from roadside_radar_seg.tools import Registry
from roadside_radar_seg.utils import (
    combine_input_embeddings_with_global_fvs,
    generate_cls_loss_targets_padded,
)

META_ARCH_REGISTRY = Registry("META_ARCH")


@META_ARCH_REGISTRY.register()
class RadarDetector(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        input_embeddings: nn.Module,
        backbone: nn.Module,
        segm_head: nn.Module,
        instance_head: nn.Module,
        feature_normalizer: nn.Module,
        model_input_fields: list,
        device: str,
        input_padding_value: int
    ) -> None:
        super().__init__()

        self.input_embeddings = input_embeddings
        self.backbone = backbone
        self.segm_head = segm_head
        self.instance_head = instance_head
        self.feature_normalizer = feature_normalizer
        self.device = device
        self.model_input_fields = model_input_fields
        self.input_padding_value = input_padding_value

        assert "index" not in self.model_input_fields

        # this is needed to simulate different behaviour in train, val and test.
        self.testing = False
        for module in self.children():
            module.testing = False

    @classmethod
    def from_config(cls, cfg):

        model_input_fields = [i for i in cfg.INPUT.INPUT_FIELDS if i != "index"]

        return {
            "input_embeddings": build_input_embeddings(cfg),
            "backbone": build_backbone(cfg),
            "segm_head": build_segm_head(cfg),
            "instance_head": build_instance_head(cfg),
            "feature_normalizer": build_feature_normalizer(cfg),
            "model_input_fields": model_input_fields,
            "device": cfg.MODEL.DEVICE,
            "input_padding_value": cfg.INPUT.INPUT_PADDING_VALUE,
        }

    def test(self, mode=True):
        self.testing = mode
        for module in self.children():
            module.testing = mode

    def sort_input_for_packed_sequence(
        self,
        input_tensors: List[torch.Tensor],
        radar_raw_point_cloud: List[np.recarray],
        gt_targets: List[dict],
    ):

        if isinstance(input_tensors, torch.Tensor):
            input_tensors = [input_tensors]

        if isinstance(radar_raw_point_cloud, np.recarray):
            radar_raw_point_cloud = [radar_raw_point_cloud]


        # Create a list of original indices: [0, 1, 2, ..., bs-1]
        indices = list(range(len(input_tensors)))

        if gt_targets is None:
            input_tensors, radar_raw_point_cloud, sorted_indices = zip(
                *sorted(
                    zip(input_tensors, radar_raw_point_cloud, indices),
                    key=lambda x: x[0].shape[0],
                    reverse=True,
                )
            )
            return input_tensors, radar_raw_point_cloud, gt_targets

        if not isinstance(gt_targets, (list, tuple)):
            gt_targets = [gt_targets]

        input_tensors, radar_raw_point_cloud, gt_targets, sorted_indices = zip(
            *sorted(
                zip(input_tensors, radar_raw_point_cloud, gt_targets, indices),
                key=lambda x: x[0].shape[0],
                reverse=True,
            )
        )

        # Calculate inverse indices
        # inv_indices[i] will tell which position in the SORTED list 
        # corresponds to the i-th element of the ORIGINAL list.
        inv_indices = [0] * len(sorted_indices)
        for sorted_pos, original_pos in enumerate(sorted_indices):
            inv_indices[original_pos] = sorted_pos

        return input_tensors, radar_raw_point_cloud, gt_targets, inv_indices

    def forward(
        self,
        batched_input_tensor: List[torch.Tensor],
        radar_raw_point_cloud: List[np.recarray],
        batched_targets: List[dict] = None,
    ):

        if not isinstance(batched_input_tensor, (list, tuple)):
            batched_input_tensor = [batched_input_tensor]

        if batched_targets is not None:
            if not isinstance(batched_targets, (list, tuple)):
                batched_targets = [batched_targets]

        bs = len(batched_input_tensor)
        # specific for inference with 1 batch size.
        # because we do not need to go back and forth between packed sequences and padded seuqences when bs = 1
        if self.testing and bs == 1:
            return self.inference_on_single_frame(
                input_tensor=batched_input_tensor,
                radar_raw_point_cloud=radar_raw_point_cloud,
            )

        # sorting for packed sequences
        batched_input_tensor, radar_raw_point_cloud, batched_targets, inv_indices = (
            self.sort_input_for_packed_sequence(
                batched_input_tensor, radar_raw_point_cloud, batched_targets
            )
        )
        # feature normalization batched_input_tensor is of type NamedBatchTensor
        batched_input_tensor = self.feature_normalizer(batched_input_tensor)
        batched_input_tensor = batched_input_tensor.to(self.device)

        # list of number of valid radar points in each frame in the batch.
        sequence_lengths = batched_input_tensor.sequence_lengths
        # input point indices (as received from raw data) of each point in each frame in batch.
        input_point_indices = batched_input_tensor["index"]

        model_input = batched_input_tensor[self.model_input_fields]
        # converting to packed sequence to ignore padded points
        model_input_packed = pack_padded_sequence(
            model_input,  # shape = [BATCH_SIZE, MAX_POINTS, NUM_FEATURES] -> [4,n,7]
            lengths=batched_input_tensor.sequence_lengths,
            batch_first=True,  # because the previous arg has `BATCH` as the first dimension.
            # enforce_sorted=False,  # If this is True, the batch must be sorted descending according to the `actual_lengths`.
        )

        # input embeddings generation
        batch_input_embeddings_packed = self.input_embeddings(model_input_packed)
        # global feature generation
        gobal_feature_vectors = self.backbone(batch_input_embeddings_packed)

        # contatenation of global features and input embeddings
        cat_features_packed, cat_features_padded = (
            combine_input_embeddings_with_global_fvs(
                batch_input_embeddings_packed,
                gobal_feature_vectors,
                self.input_padding_value,
            )
        )

        # add xyz to input embeddings
        batch_input_embeddings_padded, lengths = pad_packed_sequence(
            batch_input_embeddings_packed,
            batch_first=True,
            padding_value=self.input_padding_value,
        )

        location_fields = ["x", "y"]
        if "z" in batched_input_tensor.column_names:
            location_fields.append("z")

        batch_input_embeddings_padded = torch.cat(
            (batch_input_embeddings_padded, batched_input_tensor[location_fields]),
            dim=2,
        )

        batch_input_embeddings_packed = pack_padded_sequence(
            batch_input_embeddings_padded, lengths=lengths, batch_first=True
        )

        if not self.testing:
            # point wise ground truth class for segmentation loss calculation
            gt_cls_ids_padded = generate_cls_loss_targets_padded(
                input_point_indices,
                batched_targets,
                self.input_padding_value,
            )
            packed_gt = pack_padded_sequence(
                input=gt_cls_ids_padded, lengths=sequence_lengths, batch_first=True
            )

            cls_loss, packed_logits, padded_logits, out_features = self.segm_head(
                cat_features_packed, packed_gt
            )
            batch_avg_sim_loss, batch_radar_cluster_list = self.instance_head(
                padded_logits=padded_logits,
                batched_input=batch_input_embeddings_packed,  # cat_features_packed,
                input_point_indices=input_point_indices,
                radar_raw_point_cloud=radar_raw_point_cloud,
                batched_tagets=batched_targets,
            )
        else:
            packed_logits, padded_logits, out_features = self.segm_head(
                cat_features_packed
            )

            batch_radar_cluster_list = self.instance_head(
                padded_logits=padded_logits,
                batched_input=batch_input_embeddings_packed,  # cat_features_packed,
                input_point_indices=input_point_indices,
                radar_raw_point_cloud=radar_raw_point_cloud,
            )

        # class probabilities
        cls_predictions = F.softmax(padded_logits, dim=2)  # [B, N_max, N_cls]

        # per point cls score, and cls labels
        cls_scores, cls_labels = torch.topk(
            cls_predictions, 1
        )  # [B, N_max, 1], [B, N_max, 1]

        # cls labels of padded points is -1
        cls_labels[cls_scores.isnan()] = -1

        #if bs > 1:
        batch_radar_cluster_list_original_order = [batch_radar_cluster_list[i] for i in inv_indices]
        cls_labels_original_order = cls_labels[inv_indices]
        cls_scores_original_order = cls_scores[inv_indices]
        gt_cls_ids_padded_original_order = gt_cls_ids_padded[inv_indices]
        out_features_original_order = out_features[inv_indices]

         
        if self.testing:
            return batch_radar_cluster_list_original_order, cls_labels_original_order, cls_scores_original_order, out_features_original_order

        loss_dict = {"segm_loss": cls_loss, "sim_loss": batch_avg_sim_loss}

        return (
            loss_dict,
            batch_radar_cluster_list_original_order,
            cls_labels_original_order,
            gt_cls_ids_padded_original_order,
            out_features_original_order,
        )

    def inference_on_single_frame(
        self,
        input_tensor: Union[List, Tuple],
        radar_raw_point_cloud: np.recarray,
    ):

        assert self.testing

        if not isinstance(input_tensor, (list, tuple)):
            input_tensor = [input_tensor]

        if isinstance(radar_raw_point_cloud, np.recarray):
            radar_raw_point_cloud = [radar_raw_point_cloud]

        # feature normalization batched_input_tensor is of type NamedBatchTensor
        # [1, N_points, N_features]
        #print(input_tensor)
        input_tensor = self.feature_normalizer(input_tensor)
        input_tensor = input_tensor.to(self.device)

        # input point indices (as received from raw data) of each point in each frame in batch.
        input_point_indices = input_tensor["index"]

        model_input = input_tensor[self.model_input_fields]

        # [1, N, n_embeddings]
        input_embeddings = self.input_embeddings(model_input)

        # [1, n_fv]
        gobal_feature_vector = self.backbone(input_embeddings)

        nr_points = input_embeddings.shape[1]

        # [1, nr_points, n_fv]
        global_fv_expanded = gobal_feature_vector.expand(
            (nr_points, gobal_feature_vector.shape[-1])
        ).unsqueeze(0)

        # [1, N, n_embeddings+n_fv]
        cat_features = torch.cat((input_embeddings, global_fv_expanded), dim=2)

        # [1, N, num_class]
        pred_logits, out_features = self.segm_head(cat_features)

        # RadarCluster3dList
        # radar_cluster_list = self.instance_head(
        #     pred_logits, cat_features, input_point_indices, radar_raw_point_cloud
        # )
        location_fields = ["x", "y"]
        if "z" in input_tensor.column_names:
            location_fields.append("z")

        input_embeddings = torch.cat(
            (input_embeddings, input_tensor[location_fields]),
            dim=2,
        )
        # torch.cat
        radar_cluster_list = self.instance_head(
            pred_logits, input_embeddings, input_point_indices, radar_raw_point_cloud
        )
        # class probabilities
        cls_predictions = F.softmax(pred_logits, dim=2)  # [B, N_max, N_cls]

        # per point cls score, and cls labels
        cls_scores, cls_labels = torch.topk(
            cls_predictions, 1
        )  # [B, N_max, 1], [B, N_max, 1]

        return radar_cluster_list, cls_labels, cls_scores, out_features
