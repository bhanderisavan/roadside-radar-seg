import torch
from roadside_radar_seg.models import META_ARCH_REGISTRY


def build_model(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
