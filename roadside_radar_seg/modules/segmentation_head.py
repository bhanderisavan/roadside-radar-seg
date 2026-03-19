import torch.nn as nn
from typing import List, Union
from roadside_radar_seg.modules import MLP
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from roadside_radar_seg.tools import Registry
from roadside_radar_seg.configs import configurable
import torch
import torch.nn.functional as F

HEAD_REGISTRY = Registry("HEAD")


@HEAD_REGISTRY.register()
class RadarPointSegmentationHead(nn.Module):

    @configurable
    def __init__(
        self,
        nn_features_list: List,
        activations: List,  # len = len(nn_features_list) - 1
        normalizations: List,  # len = len(nn_features_list) - 1
        layer_type: List,  # conv or linear
        loss_name: str,
        class_wise_loss_weights: List = [],
        dropout_p: float = 0.5,
    ) -> None:
        """generates input embeddings. maps the points cloud with nn_features_list[0] features to nn_features_list[-1] features.

        Args:
            nn_features_list (List): list of number of features to map to. [input_features, ...,..., embedding_features]
            activations (List): list of activations for each layers
            normalizations (List): list of normalizations for each layers.
        """
        super().__init__()

        assert len(nn_features_list) > 2

        self.embedding_dim = nn_features_list[-1]

        if not activations:
            activations = ["leaky_relu"] * len(nn_features_list) - 1
            # activations[-1] = "softmax"

        if not normalizations:
            normalizations = ["LayerNorm"] * len(nn_features_list) - 1

        if not layer_type:
            layer_type = ["linear"] * len(nn_features_list) - 1

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
                for i in range(1, len(nn_features_list) - 2)
            ]
        )

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

        self.final_layer1 = MLP(
            in_features=nn_features_list[-3],
            out_features=nn_features_list[-2],
            activation=activations[-2],
            normalization=normalizations[-2],
            use_conv1d=layer_type[-2] == "conv",
        )

        self.final_layer2 = MLP(
            in_features=nn_features_list[-2],
            out_features=nn_features_list[-1],
            activation=activations[-1],
            normalization=normalizations[-1],
            use_conv1d=layer_type[-1] == "conv",
        )

        self.loss_name = loss_name
        self.class_wise_loss_weights = class_wise_loss_weights

    @classmethod
    def from_config(cls, cfg):

        class_loss_weights = cfg.MODEL.SEGM_HEAD.CLASS_WISE_LOSS_WEIGHTS
        if class_loss_weights:
            class_loss_weights = torch.Tensor(class_loss_weights).to(cfg.MODEL.DEVICE)

        return {
            "nn_features_list": cfg.MODEL.SEGM_HEAD.NN_LAYERS_LIST,
            "activations": cfg.MODEL.SEGM_HEAD.ACTIVATIONS_LIST,
            "normalizations": cfg.MODEL.SEGM_HEAD.NORMALIZATIONS_LIST,
            "layer_type": cfg.MODEL.SEGM_HEAD.LAYER_TYPES_LIST,
            "loss_name": cfg.MODEL.SEGM_HEAD.LOSS,
            "class_wise_loss_weights": class_loss_weights,
            "dropout_p": cfg.MODEL.SEGM_HEAD.DROPOUT,
        }

    def calculate_loss(
        self,
        logits: torch.Tensor,  # [N, num_cls] tensor
        gt: torch.Tensor,  # [N,1] long tensor
    ) -> torch.Tensor:

        if self.loss_name == "cross_entropy":

            loss = F.cross_entropy(
                input=logits,  # [N, num_cls] tensor
                target=gt.long(),  # [N,1] long tensor
                weight=self.class_wise_loss_weights,  # [num_cls] tensor
                reduction="mean",
            )
        else:
            raise NotImplementedError()

        return loss

    def forward(
        self,
        merged_features: Union[PackedSequence, torch.Tensor],
        packed_gt: PackedSequence = None,
    ):

        # if input type is tensor then batch size must be 1.
        if isinstance(merged_features, torch.Tensor):
            assert merged_features.shape[0] == 1

        if self.testing and isinstance(merged_features, torch.Tensor):
            out = self.mlp(merged_features)
            out = self.dropout1(out)
            out_features = self.final_layer1(out)
            out = self.dropout2(out_features)
            out = self.final_layer2(out)
            return out, out_features

        assert isinstance(merged_features, PackedSequence)

        ip, batch_sizes, sorted_indices, unsorted_indices = merged_features
        out = self.mlp(ip)
        out = self.dropout1(out)
        out_features = self.final_layer1(out)
        out = self.dropout2(out_features)
        out = self.final_layer2(out)

        packed_logits = PackedSequence(
            out,  # [N ,num_classes]
            batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )

        # padding of infinity will give nan after softmax - easier to separate
        padded_logits, _ = pad_packed_sequence(
            packed_logits,
            batch_first=True,
            padding_value=float("-inf"),
            total_length=None,
        )

        packed_features = PackedSequence(
            out_features,  # [N ,num_classes]
            batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )

        # padding of infinity will give nan after softmax - easier to separate
        padded_features, _ = pad_packed_sequence(
            packed_features,
            batch_first=True,
            padding_value=float("-inf"),
            total_length=None,
        )


        if self.testing:
            return packed_logits, padded_logits, padded_features

        cls_loss = self.calculate_loss(packed_logits.data, packed_gt.data)

        return cls_loss, packed_logits, padded_logits, padded_features


def build_segm_head(cfg):
    segm_head_name = cfg.MODEL.SEGM_HEAD.NAME
    return HEAD_REGISTRY.get(segm_head_name)(cfg)
