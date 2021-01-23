import gym
import numpy as np
import torch

from rlcode.data.buffer import create_buffer
from rlcode.data.experience import EpisodeExperienceSource, NStepExperienceSource
from rlcode.policy.dqn import DQNPolicy
from rlcode.policy.transformed_dqn import TransformedDQNPolicy
from rlcode.utils.net import QNet
from rlcode.utils.trainer import Trainer


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


def process_cfg(cfg: dict) -> dict:
    seed = cfg["seed"]
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def make_env() -> gym.Env:
        env = cfg["env_fn"](**cfg["env"])
        if seed is not None:
            env.seed(seed)
        return env

    cfg["make_env"] = make_env

    env = make_env()
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_n = env.observation_space["obs"].shape[0]
    else:
        obs_n = env.observation_space.shape[0]
    act_n = env.action_space.n
    env.close()

    cfg["policy"]["network"]["obs_n"] = obs_n
    cfg["policy"]["network"]["act_n"] = act_n

    return cfg


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
