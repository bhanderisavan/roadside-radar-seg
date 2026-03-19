from .dataset.radar_dataset import RadarDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from functools import partial

def collate_fun(batch):
    return tuple(zip(*batch))


def _seed_worker(worker_id, num_workers):
    worker_seed = torch.initial_seed() % 2**num_workers
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_train_loader(cfg):



    dataset = RadarDataset(cfg, "train")

    loader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=cfg.DATALOADER.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=collate_fun,
        drop_last=cfg.DATALOADER.DROP_LAST,
        worker_init_fn=partial(_seed_worker, num_workers=cfg.DATALOADER.NUM_WORKERS),
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
    )

    return loader


def build_val_loader(cfg):

    dataset = RadarDataset(cfg, "val")

    loader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=cfg.DATALOADER.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=collate_fun,
        drop_last=cfg.DATALOADER.DROP_LAST,
        worker_init_fn=partial(_seed_worker, num_workers=cfg.DATALOADER.NUM_WORKERS),
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
    )

    return loader
