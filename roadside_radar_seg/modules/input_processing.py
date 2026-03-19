import torch.nn as nn
from typing import List, Union
from roadside_radar_seg.modules import MLP
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from roadside_radar_seg.configs import configurable
from roadside_radar_seg.tools import Registry
import torch

INPUT_EMBEDDINGS_REGISTRY = Registry("INPUT_EMBEDDINGS")


@INPUT_EMBEDDINGS_REGISTRY.register()
class MLPInputEmbeddings(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        nn_features_list: List,
        activations: List = [],  # len = len(nn_features_list) - 1
        normalizations: List = [],  # len = len(nn_features_list) - 1
        layer_type: List = [],  # conv or linear
    ) -> None:
        """generates input embeddings. maps the points cloud with nn_features_list[0] features to nn_features_list[-1] features.

        Args:
            nn_features_list (List): list of number of features to map to. [input_features, ...,..., embedding_features]
            activations (List): list of activations for each layers
            normalizations (List): list of normalizations for each layers.
        """
        super().__init__()

        assert len(nn_features_list) > 1

        self.embedding_dim = nn_features_list[-1]

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
            "nn_features_list": cfg.MODEL.INPUT_PROCESSING.NN_LAYERS_LIST,
            "activations": cfg.MODEL.INPUT_PROCESSING.ACTIVATIONS_LIST,
            "normalizations": cfg.MODEL.INPUT_PROCESSING.NORMALIZATIONS_LIST,
            "layer_type": cfg.MODEL.INPUT_PROCESSING.LAYER_TYPES_LIST,
        }

    def forward(self, input_data: Union[PackedSequence, torch.Tensor]):

        # if input type is tensor then batch size must be 1.
        if isinstance(input_data, torch.Tensor):
            assert input_data.shape[0] == 1

        if self.testing and isinstance(input_data, torch.Tensor):
            return self.mlp(input_data)

        assert isinstance(input_data, PackedSequence)

        ip, batch_sizes, sorted_indices, unsorted_indices = input_data

        out = self.mlp(ip)

        batch_input_embeddings_packed = PackedSequence(
            out,
            batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )

        # batch_input_embeddings_padded, _ = pad_packed_sequence(
        #     batch_input_embeddings_packed, batch_first=True, padding_value=0.0
        # )

        return batch_input_embeddings_packed


def build_input_embeddings(cfg):
    input_embeddings_name = cfg.MODEL.INPUT_PROCESSING.NAME
    return INPUT_EMBEDDINGS_REGISTRY.get(input_embeddings_name)(cfg)
