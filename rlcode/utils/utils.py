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
