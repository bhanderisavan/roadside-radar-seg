import torch.nn as nn
import torch.nn.functional as F
import math

class MultiplicativeSimilarity(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, query, key):
        scale_factor = 1 / math.sqrt(query.size(-1))
        return query @ key.transpose(-2, -1) * scale_factor


class AdditiveSimilarity(nn.Module):

    def __init__(self, input_size) -> None:
        super().__init__()
        self.v = nn.Linear(input_size, input_size, bias=False)

        nn.init.kaiming_normal_(self.v.weight)

    def forward(self, query, key):
        features = F.tanh(query + key)
        return self.v(features)
