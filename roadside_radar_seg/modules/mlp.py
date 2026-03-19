import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torch


class MLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "",
        normalization: str = "",
        use_conv1d: bool = False,  # use conv1d instead of linear layers
    ) -> None:
        """a multi layer perceptron module with linear -> activation -> norm sequence operations.

        Args:
            in_features (int): _description_
            out_features (int): _description_
            activation (str, optional): _description_. Defaults to "leaky_relu".
            normalization (str, optional): _description_. Defaults to "LayerNorm".
            use_conv1d (bool, optional): _description_. Defaults to False.
        """
        # activation - relu, leaky_relu, softmax, sigmoid
        # normalization - bn or ln
        super().__init__()

        self.activation = activation.lower()
        assert self.activation in [
            "relu",
            "leaky_relu",
            "softmax",
            "sigmoid",
            "silu",
            "gelu",
            "",
        ], f"invalid activation function: {self.activation}"

        self.normalization = normalization.lower()
        assert self.normalization in [
            "layernorm",
            "batchnorm",
            "",
        ], f"invalid activation function: {self.normalization}"

        self.out_features = out_features

        self.use_conv1d = use_conv1d

        if self.use_conv1d:
            self.nn_layer = nn.Conv1d(in_features, out_features, kernel_size=1)
        else:
            self.nn_layer = nn.Linear(in_features, out_features)

        self.activation_func = self._get_activation_func()

        self.norm = self._get_normalization_func()

        nn.init.kaiming_normal_(self.nn_layer.weight)
        nn.init.constant_(self.nn_layer.bias, 0)

    def _get_normalization_func(self):

        if self.normalization == "layernorm":
            return nn.LayerNorm(self.out_features)
        elif self.normalization == "batchnorm":
            return nn.BatchNorm1d(self.out_features)
        elif self.normalization == "":
            return None
        else:
            raise NotImplementedError()

    def _get_activation_func(self):

        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU()
        elif self.activation == "softmax":
            return nn.Softmax()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        elif self.activation == "silu":
            return nn.SiLU()
        elif self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "":
            return None
        else:
            raise NotImplementedError()

    def forward(self, input_tensor: torch.Tensor):  # shape = [N,M]

        is_batched = input_tensor.ndim == 3
        if is_batched:
            assert input_tensor.shape[0] == 1

        if self.use_conv1d:

            if is_batched:
                input_tensor = input_tensor.squeeze()

            out = self.nn_layer(input_tensor.permute(1, 0))
            if self.norm is not None:
                out = self.norm(out.permute(1, 0))
            else:
                out = out.permute(1, 0)
            if self.activation_func is not None:
                out = self.activation_func(out)

            if is_batched:
                out = out.unsqueeze(0)
        else:
            out = self.nn_layer(input_tensor)
            if self.norm is not None:
                out = self.norm(out)
            if self.activation_func is not None:
                out = self.activation_func(out)

        return out
