import torch.nn as nn
import torch
import torch.nn.functional as F
from roadside_radar_seg.modules import AdditiveSimilarity, MultiplicativeSimilarity
from roadside_radar_seg.configs import configurable


class SelfAttention(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        input_dim: int,
        similarity_type: str = "multiplicative",
        embed_values=False,
    ) -> None:
        super().__init__()

        self.queries_fc = nn.Linear(input_dim, input_dim)
        self.keys_fc = nn.Linear(input_dim, input_dim)

        self.embed_values = embed_values

        if self.embed_values:
            self.values_fc = nn.Linear(input_dim, input_dim)

        assert similarity_type in ["multiplicative", "additive"]

        if similarity_type == "multiplicative":
            self.similarity = MultiplicativeSimilarity()
        else:
            self.similarity = AdditiveSimilarity(input_dim)

        for c in self.children():
            if isinstance(c, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(c.weight)
                nn.init.constant_(c.bias, 0)

    @classmethod
    def from_config(cls, cfg, attn_input_shape):
        return {
            "input_dim": attn_input_shape,
            "similarity_type": cfg.MODEL.INSTANCE_HEAD.ATTENTION.SIMILARITY_TYPE,
            "embed_values": cfg.MODEL.INSTANCE_HEAD.ATTENTION.EMBED_VALUES,
        }

    def forward(self, input_tensor):

        Q = self.queries_fc(input_tensor)
        K = self.keys_fc(input_tensor)

        scores_logits = self.similarity(Q, K)
        scores = torch.sigmoid(scores_logits)

        if self.embed_values:
            V = self.values_fc(input_tensor)
            return torch.matmul(scores, V), scores

        return torch.matmul(scores, input_tensor), scores_logits


def build_self_attention(cfg):

    if cfg.MODEL.INSTANCE_HEAD.NN_LAYERS_LIST:
        attn_input_shape = cfg.MODEL.INSTANCE_HEAD.NN_LAYERS_LIST[-1]
    else:
        attn_input_shape = (
            cfg.MODEL.INPUT_PROCESSING.NN_LAYERS_LIST
            + cfg.MODEL.BACKBONE.NN_LAYERS_LIST[-1]
        )
    return SelfAttention(cfg, attn_input_shape)
