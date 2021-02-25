import gym
import numpy as np

from rlcode.data.buffer import ReplayBuffer
from rlcode.data.collector import Collector


def collect_test(test_trajectory, test_buffer):
    env = gym.make("Pendulum-v0")

    def infer(*args, **kwargs):
        return env.action_space.sample()

    buf = (
        ReplayBuffer(10000, 1, env.observation_space, env.action_space, True)
        if test_buffer
        else None
    )

    nstep = 100
    collector = Collector(infer, env, buf)

    if test_trajectory:
        dat = collector.collect_trajectory()
        assert dat.mask is None
        assert dat.done[-1]
        assert not np.any(dat.done[:-1])

        if buf is not None:
            bat = buf.get_all(True)
            assert bat.mask is None

            assert dat.obs.shape == bat.obs.shape
            assert dat.act.shape == bat.act.shape
            assert dat.rew.shape == bat.rew.shape
            assert dat.done.shape == bat.done.shape
            assert dat.next_obs.shape == bat.next_obs.shape

            assert np.all(np.isclose(dat.obs, bat.obs))
            assert np.all(np.isclose(dat.act, bat.act))
            assert np.all(np.isclose(dat.rew, bat.rew))
            assert np.all(np.isclose(dat.done, bat.done))
            assert np.all(np.isclose(dat.next_obs, bat.next_obs))

    else:
        dat = collector.collect_nstep(nstep)
        assert dat.mask is None
        assert (
            nstep
            == dat.obs.shape[0]
            == dat.act.shape[0]
            == dat.rew.shape[0]
            == dat.done.shape[0]
            == dat.next_obs.shape[0]
        )


def test_collect_trajectory():
    collect_test(True, True)
    collect_test(True, False)


def test_collect_nstep():
    collect_test(False, True)
    collect_test(False, False)


if __name__ == "__main__":
    test_collect_trajectory()
    test_collect_nstep()
