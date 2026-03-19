import torch.nn as nn
from typing import List, Tuple
import torch
from roadside_radar_seg.structures import NamedBatchTensor
from torch.nn.utils.rnn import pad_sequence
from roadside_radar_seg.configs import configurable
from roadside_radar_seg.utils import convert_cfgnode_to_dict


class FeatureNormalizer(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        input_field_names: List[str],
        normalization_type: str,
        feature_dim: int,
        input_padding_value: int = -1,
        fieldwise_minmax: dict = None,
    ):
        """expects that each tensor in the list has input_field_names + 1 coumns. the first column being the index field.
        Args:
            input_field_names (List[str]): list of column names for the input tensor.
            fieldwise_minmax (dict):  dictionary with keys = fields, and values = min max values per field. Defaults to None.

        """
        super().__init__()

        self.fieldwise_minmax = fieldwise_minmax
        self.normalization_type = normalization_type

        # must duplicate the list, otherwise it will get mutated.
        self.input_field_names = input_field_names[:]
        self.feature_dim = feature_dim
        self.input_padding_value = input_padding_value

        if "index" not in self.input_field_names:
            self.input_field_names.insert(0, "index")

    @classmethod
    def from_config(cls, cfg):

        fieldwise_minmax = None
        if cfg.INPUT.NORMALIZATION_METHOD == "minmax":
            fieldwise_minmax = cfg.INPUT.FEATURE_WISE_MIN_MAX
            fieldwise_minmax = convert_cfgnode_to_dict(fieldwise_minmax)

        return {
            "fieldwise_minmax": fieldwise_minmax,
            "input_field_names": cfg.INPUT.INPUT_FIELDS,
            "normalization_type": cfg.INPUT.NORMALIZATION_METHOD,
            "input_padding_value": cfg.INPUT.INPUT_PADDING_VALUE,
            "feature_dim": cfg.INPUT.FEATURE_DIM,
        }

    def forward(
        self, input_point_clouds_batch: Tuple[torch.Tensor]
    ) -> NamedBatchTensor:
        """pads the input tensor and creates one batch tensor of type BatchTensor.

        Args:
            input_point_clouds_batch (Tuple[torch.Tensor]): tuple of tensors, one for each input frame in the batch.

        Returns:
            NamedBatchTensor
        """
        lengths = [i.shape[0] for i in input_point_clouds_batch]  # len = batch_size
        # normalize here
        normalized_input_list = [self.normalize(i) for i in input_point_clouds_batch]

        if len(input_point_clouds_batch) == 1:
            # [1, N_pts, N_features]
            batch_tensor = normalized_input_list[0].unsqueeze(0)
        else:
            batch_tensor = pad_sequence(
                normalized_input_list,
                batch_first=True,
                padding_value=self.input_padding_value,
            )  # shape = [batch_size, N, num_features]. N = max(lengths)

        return NamedBatchTensor(
            tensor=batch_tensor,
            column_names=self.input_field_names,
            sequence_lengths=lengths,
            feature_dim=self.feature_dim,
        )

    def normalize(self, input_tensor: torch.Tensor) -> torch.Tensor:

        if self.normalization_type == "minmax":

            assert isinstance(
                self.fieldwise_minmax, dict
            ), f"pass a dict for fieldwise_minmax"

            normalized_tensor = input_tensor.clone()
            for f_idx, f_name in enumerate(self.input_field_names):

                # do not normalize index field.
                if f_name == "index":
                    continue

                assert f_name in self.fieldwise_minmax.keys()

                f_min = self.fieldwise_minmax[f_name]["min"]
                f_max = self.fieldwise_minmax[f_name]["max"]

                # (feature - feature_min) / (feature_max - feature_min)
                normalized_tensor[:, f_idx] = (normalized_tensor[:, f_idx] - f_min) / (
                    f_max - f_min
                )
        else:
            raise NotImplementedError(
                f"currently {self.normalization_type} method is not supported."
            )

        return normalized_tensor


def build_feature_normalizer(cfg):
    return FeatureNormalizer(cfg)
