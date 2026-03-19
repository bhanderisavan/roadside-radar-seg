from torch.optim import lr_scheduler
from torch import optim


def build_optimizer(cfg, model):
    name = cfg.SOLVER.OPTIMIZER.NAME

    if name == "Adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif name == "AdamW":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    elif name == "SGD":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    else:
        raise NotImplementedError(f"unknown optimizer name : {name}")

    return optimizer


def build_lr_scheduler(cfg, optimizer):

    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "MultiStepLR":
        milestones = [
            x for x in cfg.SOLVER.LR_SCHEDULER_MILESTONES if x <= cfg.SOLVER.MAX_EPOCHS
        ]
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=cfg.SOLVER.LR_SCHEDULER_GAMMA,
        )
    elif name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.SOLVER.MAX_EPOCHS,
            eta_min=cfg.SOLVER.BASE_LR_END,
        )
    elif name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=cfg.SOLVER.LR_SCHEDULER_PATIENCE,
            factor=cfg.SOLVER.LR_SCHEDULER_GAMMA,
            min_lr=cfg.SOLVER.BASE_LR_END,
        )
    else:
        raise NotImplementedError(f"unknown lr scheduler {name}.")

    return scheduler
