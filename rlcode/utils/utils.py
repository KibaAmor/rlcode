# -*- coding: utf-8 -*-
import gym
import numpy as np
import torch


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

    env = make_env()
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_space = env.observation_space["obs"]
    else:
        obs_space = env.observation_space
    act_space = env.action_space
    env.close()

    cfg["make_env"] = make_env
    cfg["env_space"] = dict(
        obs_space=obs_space,
        act_space=act_space,
    )

    return cfg
