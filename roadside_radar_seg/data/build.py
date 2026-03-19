from .dataset.radar_dataset import RadarDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random


def build_train_loader(cfg):

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**cfg.DATALOADER.NUM_WORKERS
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset = RadarDataset(cfg, "train")

    loader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=cfg.DATALOADER.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=lambda batch: tuple(zip(*batch)),
        drop_last=cfg.DATALOADER.DROP_LAST,
        worker_init_fn=_seed_worker,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
    )

    return loader


def build_val_loader(cfg):

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**cfg.DATALOADER.NUM_WORKERS
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset = RadarDataset(cfg, "val")

    loader = DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        shuffle=cfg.DATALOADER.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=lambda batch: tuple(zip(*batch)),
        drop_last=cfg.DATALOADER.DROP_LAST,
        worker_init_fn=_seed_worker,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
    )

    return loader
