import torch
import torch.nn as nn
from roadside_radar_seg.modules import MLP
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from typing import List, Union
from roadside_radar_seg.configs import configurable
from roadside_radar_seg.tools import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")


@BACKBONE_REGISTRY.register()
class MLPBackbone(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        nn_features_list: List,
        activations: List = [],  # len = len(nn_features_list) - 1
        normalizations: List = [],  # len = len(nn_features_list) - 1
        layer_type: List = [],  # conv or linear
    ) -> None:
        super().__init__()

        assert len(nn_features_list) > 1

        self.embedding_dim = nn_features_list[1]
        self.fv_size = nn_features_list[-1]

        if not activations:
            activations = ["leaky_relu"] * len(nn_features_list) - 1

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
                for i in range(1, len(nn_features_list))
            ]
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "nn_features_list": cfg.MODEL.BACKBONE.NN_LAYERS_LIST,
            "activations": cfg.MODEL.BACKBONE.ACTIVATIONS_LIST,
            "normalizations": cfg.MODEL.BACKBONE.NORMALIZATIONS_LIST,
            "layer_type": cfg.MODEL.BACKBONE.LAYER_TYPES_LIST,
        }

    def forward(self, input_data: Union[PackedSequence, torch.Tensor]):

        # if input type is tensor then batch size must be 1.
        if isinstance(input_data, torch.Tensor):
            assert input_data.shape[0] == 1

        if self.testing and isinstance(input_data, torch.Tensor):
            out = self.mlp(input_data)
            feature_vectors = torch.max(out, 1)[0]
            return feature_vectors

        assert isinstance(input_data, PackedSequence)

        ip, batch_sizes, sorted_indices, unsorted_indices = input_data

        out = self.mlp(ip)

        features_packed = PackedSequence(
            out,
            batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )

        # convert packed input to padded tensor of shape [batch, num_points, num_features] with padding value = -inf
        # -inf is used to ignore the padded points in max pooling
        # ip padded shape = [batch, N, 1024]. N = max(sequence_lengths)
        # we can also pad the N with total_length. (i.e 200.)
        features_padded, _ = pad_packed_sequence(
            features_packed,
            batch_first=True,
            padding_value=float("-inf"),
            total_length=None,
        )

        feature_vectors = torch.max(features_padded, 1)[0]  # [batch_size,1024]

        return feature_vectors


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    return BACKBONE_REGISTRY.get(backbone_name)(cfg)
