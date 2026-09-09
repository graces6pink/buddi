import torch
from loguru import logger as guru

def build_optimizer(cfg, optimizer_type, params):
    """
    Build optimizer from config.
    Parameters
    ----------
    cfg: cfg
        The configuration of the optimizer.
    optimizer_type: str
        The type of the optimizer to build.
    params: list
        The parameters to optimize.
    """
    
    # setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg.adam.lr, 
            weight_decay=cfg.adam.weight_decay
        )
    elif optimizer_type.lower() == 'lbfgs':
        optimizer = torch.optim.LBFGS(
            params=params,
            lr=cfg.lbfgs.lr
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(cfg, optimizer):
    """
    Build an LR scheduler from config. Returns None if no scheduler is configured.
    Parameters
    ----------
    cfg: cfg
        The configuration of the optimizer (same cfg passed to build_optimizer).
    optimizer: torch.optim.Optimizer
        The optimizer to attach the scheduler to.
    """

    if cfg.scheduler.type == 'none':
        return None
    elif cfg.scheduler.type == 'reduce_lr_on_plateau':
        p = cfg.scheduler.reduce_lr_on_plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=p.mode, factor=p.factor, patience=p.patience, min_lr=p.min_lr
        )
    else:
        raise NotImplementedError

    return scheduler