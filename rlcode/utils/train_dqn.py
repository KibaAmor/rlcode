import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlcode.data.buffer import create_buffer
from rlcode.data.experience import EpisodeExperienceSource, NStepExperienceSource
from rlcode.policy.dqn import DQNPolicy
from rlcode.utils.net import QNet
from rlcode.utils.trainer import Trainer


class DQN(DQNPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def network(self) -> torch.nn.Module:
        return self.net

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.optim

    def create_network_optimizer(self, network: dict, optim: dict) -> None:
        self.net = QNet(**network).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), **optim)


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
    buffer = create_buffer(**cfg["buffer"])

    train_env = cfg["make_env"]()
    train_src = NStepExperienceSource(policy, train_env, cfg["exp_per_collect"], buffer)

    test_env = cfg["make_env"]()
    test_src = EpisodeExperienceSource(policy, test_env)

    writer = SummaryWriter(**cfg["writer"])
    if cfg["warmup_size"] > 0:
        batch = train_src.collect(nstep=cfg["warmup_size"])
        dummy = torch.FloatTensor(batch.obss)
        writer.add_graph(policy, dummy)

    trainer = Trainer(policy, train_src, test_src, writer)

    rew = trainer.train(**cfg["train"])

    train_env.close()
    test_env.close()

    return rew
