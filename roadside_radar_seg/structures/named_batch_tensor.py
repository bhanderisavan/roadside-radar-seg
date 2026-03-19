import torch
from typing import List, Union


class NamedBatchTensor:
    def __init__(
        self,
        tensor: torch.Tensor,
        column_names: List[str],
        sequence_lengths: List[int],
        feature_dim: int = 2,
    ) -> None:
        """a custom class that supports tensor indexing with column names.


        Args:
            tensor (torch.Tensor): tensor data of shape [B,N,M] each sample has a different number of points, and therefore is padded to the max points in the current batch.
            column_names (List[str]): list of column names for the tensor.
            sequence_lengths (List[int]): len = tensor.shape[0] number of valid points in each sample.
            feature_dim (int): which dimension of the tensor represents the features. Defaults to 2. meaning that tensor is [batch_size, num_points, num_features].
        """
        self.tensor = tensor  # tensor with actual data N*M
        self.column_names = column_names  # field names for each column of the tensor.
        self.shape = self.tensor.shape
        self.sequence_lengths = torch.Tensor(sequence_lengths)  # len = batch size.
        self.feature_dim = int(feature_dim)

        assert self.feature_dim in [1, 2]

        assert (
            len(self.sequence_lengths) == self.tensor.shape[0]
        ), "number of element in sequence lenght must be = tensor.shape[0]"

    def __getitem__(self, column_names: Union[str, list]):
        """indexing the tensor using column names."""

        if isinstance(column_names, str):
            assert column_names in self.column_names
            if self.feature_dim == 1:

                return self.tensor[:, self.column_names.index(column_names)].clone()

            return self.tensor[:, :, self.column_names.index(column_names)].clone()
        if not isinstance(column_names, list):
            raise NotImplementedError(
                "only supports string or list of strings as index."
            )

        for i in column_names:
            assert isinstance(i, str)
            assert i in self.column_names
        col_indices = [self.column_names.index(i) for i in column_names]

        if self.feature_dim == 2:

            return self.tensor[
                :, :, col_indices
            ].clone()  # self.tensor[:, col_indices, :]
        # indexing along dim 1
        return self.tensor[:, col_indices, :].clone()  # self.tensor[:, :, col_indices]

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self
