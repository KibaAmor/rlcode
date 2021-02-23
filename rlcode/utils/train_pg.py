# -*- coding: utf-8 -*-
import torch

from rlcode.data.experience import EpisodeExperienceSource
from rlcode.policy.pg import PGPolicy
from rlcode.utils.net import PNet
from rlcode.utils.trainer import Trainer
from rlcode.utils.utils import process_cfg


class FCPG:
    @property
    def network(self) -> torch.nn.Module:
        return self.net

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optim

    def create_network_optimizer(self, network: dict, optim: dict) -> None:
        self.net = PNet(**network).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), **optim)


class PG(FCPG, PGPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def train_pg(cfg: dict, cls: PGPolicy) -> float:
    cfg = process_cfg(cfg)

    policy = cls(**cfg["policy"])

    train_env = cfg["make_env"]()
    train_src = EpisodeExperienceSource(policy, train_env, **cfg["train_src"])

    test_env = cfg["make_env"]()
    test_src = EpisodeExperienceSource(policy, test_env, **cfg["test_src"])

    trainer = Trainer(policy, train_src, test_src, **cfg["trainer"])
    rew = trainer.train(**cfg["train"])

    train_env.close()
    test_env.close()

    return rew
