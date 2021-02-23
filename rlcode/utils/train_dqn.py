# -*- coding: utf-8 -*-
import torch

from rlcode.data.buffer import create_buffer
from rlcode.data.experience import EpisodeExperienceSource, NStepExperienceSource
from rlcode.policy.dqn import DQNPolicy
from rlcode.policy.transformed_dqn import TransformedDQNPolicy
from rlcode.utils.net import QNet
from rlcode.utils.trainer import Trainer
from rlcode.utils.utils import process_cfg


class FCDQN:
    @property
    def network(self) -> torch.nn.Module:
        return self.net

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optim

    def create_network_optimizer(self, network: dict, optim: dict) -> None:
        self.net = QNet(**network).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), **optim)


class DQN(FCDQN, DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TransformedDQN(FCDQN, TransformedDQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def train_dqn(cfg: dict, cls: DQNPolicy) -> float:
    cfg = process_cfg(cfg)

    policy = cls(**cfg["policy"])

    train_env = cfg["make_env"]()
    buffer = create_buffer(**cfg["buffer"])
    train_src = NStepExperienceSource(policy, train_env, buffer, **cfg["train_src"])

    test_env = cfg["make_env"]()
    test_src = EpisodeExperienceSource(policy, test_env, **cfg["test_src"])

    trainer = Trainer(policy, train_src, test_src, **cfg["trainer"])
    rew = trainer.train(**cfg["train"])

    train_env.close()
    test_env.close()

    return rew
