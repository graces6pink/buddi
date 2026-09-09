import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass 
class Adam:
    lr: float = 1.0
    weight_decay: float = 0.0

@dataclass
class LBFGS:
    lr: float = 1.0

@dataclass
class ReduceLROnPlateau:
    mode: str = 'min'
    factor: float = 0.5
    patience: int = 2
    min_lr: float = 1e-6

@dataclass
class Scheduler:
    type: str = 'none'  # 'none' | 'reduce_lr_on_plateau'
    reduce_lr_on_plateau: ReduceLROnPlateau = ReduceLROnPlateau()

@dataclass
class Optimizer:
    type: str = 'adam'
    adam: Adam = Adam()
    lbfgs: LBFGS = LBFGS()
    scheduler: Scheduler = Scheduler()