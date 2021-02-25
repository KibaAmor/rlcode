# -*- coding: utf-8 -*-
import torch as th

from rlcode.data.buffer import create_buffer
from rlcode.data.collector import Collector
from rlcode.policy.dqn import DQNPolicy
from rlcode.policy.transformed_dqn import TransformedDQNPolicy
from rlcode.utils.net import QNet
from rlcode.utils.trainer import OffPolicyTrainer, OnPolicyTrainer
from rlcode.utils.utils import process_cfg


class FCDQN:
    @property
    def net(self) -> th.nn.Module:
        return self.__net

    @property
    def optim(self) -> th.optim.Optimizer:
        return self.__optim

    def build(self, net: dict, optim: dict) -> None:
        self.__net = QNet(obs_space=self.obs_space, act_space=self.act_space, **net).to(self.device)
        self.__optim = th.optim.Adam(self.net.parameters(), **optim)


class DQN(FCDQN, DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TransformedDQN(FCDQN, TransformedDQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def train_dqn(cfg: dict, cls: DQNPolicy) -> float:
    cfg = process_cfg(cfg)

    policy = cls(**cfg["env_space"], **cfg["policy"])

    train_env = cfg["make_env"]()
    train_buffer = create_buffer(**cfg["env_space"], **cfg["train_buffer"])
    train_collector = Collector(policy.infer, train_env, train_buffer, **cfg["train_collector"])

    test_env = cfg["make_env"]()
    test_collector = Collector(policy.infer, test_env, **cfg["test_collector"])

    trainer_cls = OffPolicyTrainer if cfg["off_policy_trainer"] else OnPolicyTrainer
    trainer = trainer_cls(policy, train_collector, test_collector, **cfg["trainer"])
    rew = trainer.train(**cfg["train"])

    train_env.close()
    test_env.close()

    return rew
