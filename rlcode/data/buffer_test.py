# -*- coding: utf-8 -*-

import gym
import numpy as np
from pytest import approx

from rlcode.data.buffer import (
    PrioReplayBuffer,
    ReplayBuffer,
    SumSegmentTree,
    get_act_dim,
    get_obs_shape,
)


def test_SumSegmentTree():
    data = np.random.rand(50) * 100

    st = SumSegmentTree(len(data))
    for i, v in enumerate(data):
        st[i] = v

    assert len(data) == len(st)
    assert all([a == approx(b) for a, b in zip(data, st)])
    assert st.reduce() == approx(sum(st))

    for left in range(len(st)):
        for right in range(left + 1, len(st)):
            assert st.reduce(left, right) == approx(sum(data[left:right]))

    def get_prefix_sum_index(st, value):
        for i in range(len(st)):
            if st[i] >= value:
                return i
            value -= st[i]
        assert False

    for v in range(int(sum(st))):
        assert get_prefix_sum_index(st, v) == st.get_prefix_sum_index(v)


def buffer_test(env_name, buffer_size, batch_size, save_next_obs, has_mask, test_prio):
    env = gym.make(env_name)
    obs_shape = get_obs_shape(env.observation_space)
    act_dim = get_act_dim(env.action_space)

    if test_prio:
        buf = PrioReplayBuffer(
            1,  # for test perpose
            -1,
            buffer_size,
            batch_size,
            env.observation_space,
            env.action_space,
            save_next_obs=save_next_obs,
            has_mask=has_mask,
        )
    else:
        buf = ReplayBuffer(
            buffer_size,
            batch_size,
            env.observation_space,
            env.action_space,
            save_next_obs=save_next_obs,
            has_mask=has_mask,
        )
    obs = env.reset()

    def step():
        nonlocal obs
        act = env.action_space.sample()
        next_obs, rew, done, _ = env.step(act)
        fake_mask = np.random.binomial(1, 0.5) > 0
        if done:
            next_obs = env.reset()
        buf.add(obs, act, rew, done, next_obs, fake_mask)
        obs = next_obs

    batch_obs_shape = (batch_size,) + obs_shape
    batch_act_shape = (batch_size, act_dim)
    batch_scalar_shape = (batch_size,)

    def test(has_next_obs):
        if len(buf) < batch_size:
            return

        bat = buf.sample(has_next_obs=has_next_obs)
        assert bat.obs.shape == batch_obs_shape
        assert bat.act.shape == batch_act_shape
        assert bat.rew.shape == batch_scalar_shape
        assert bat.done.shape == batch_scalar_shape
        if has_next_obs:
            assert bat.next_obs.shape == batch_obs_shape
        else:
            assert bat.next_obs is None

        if has_mask:
            assert bat.mask.shape == batch_act_shape
        else:
            assert bat.mask is None

        if test_prio:
            assert bat.index.shape == batch_scalar_shape
            assert bat.weight.shape == batch_scalar_shape

            # remove replicated index
            index = np.unique(bat.index)
            new_weight = np.abs(np.random.rand(len(index))) - np.finfo(float).eps.item()
            buf.update(index, new_weight)
            updated_weight = buf.get(index, has_next_obs).weight * buf.min_prio
            assert updated_weight == approx(new_weight)

        all_data = buf.get_all(True)
        next_obs = all_data.next_obs
        obs = all_data.obs
        count = 0
        for i in range(len(buf) - 1):
            if not np.all(np.isclose(next_obs[i], obs[i + 1])):
                count += 1
        assert count <= 1

    for i in range(buffer_size):
        assert i == len(buf)
        step()
        test(True)
        test(False)

    for i in range(buffer_size):
        assert buffer_size == len(buf)
        step()
        test(True)
        test(False)


def replay_buffer_test(env_name):
    buffer_test(env_name, 20, 6, True, True, False)
    buffer_test(env_name, 20, 6, True, False, False)
    buffer_test(env_name, 20, 6, False, True, False)
    buffer_test(env_name, 20, 6, False, False, False)


def prio_replay_buffer_test(env_name):
    buffer_test(env_name, 20, 6, True, True, True)
    buffer_test(env_name, 20, 6, True, False, True)
    buffer_test(env_name, 20, 6, False, True, True)
    buffer_test(env_name, 20, 6, False, False, True)


def test_ReplayBuffer_with_discrete():
    env_name = "CliffWalking-v0"
    replay_buffer_test(env_name)
    prio_replay_buffer_test(env_name)


def test_ReplayBuffer_with_continues():
    env_name = "Pendulum-v0"
    replay_buffer_test(env_name)
    prio_replay_buffer_test(env_name)


if __name__ == "__main__":
    test_SumSegmentTree()
    test_ReplayBuffer_with_discrete()
    test_ReplayBuffer_with_continues()
