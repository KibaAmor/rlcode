# -*- coding: utf-8 -*-
import gym

from rlcode.utils.train_dqn import TransformedDQN, train_dqn


def get_cfg() -> dict:
    cfg = dict(
        seed=None,
        env_fn=gym.make,
        env=dict(
            id="CartPole-v0",
        ),
        policy=dict(
            target_update_freq=500,
            gamma=0.99,
            device=None,
            net=dict(
                layer_num=3,
                hidden_size=256,
                adv_layer_num=2,
                val_layer_num=2,
                activation="ReLU",
            ),
            optim=dict(
                lr=6e-5,
            ),
        ),
        train_buffer=dict(
            alpha=0.5,
            beta=0.4,
            buffer_size=10000,
            batch_size=128,
            save_next_obs=False,
            has_mask=False,
            always_copy=False,
        ),
        train_collector=dict(
            max_episode_step=1000,
        ),
        test_collector=dict(
            max_episode_step=None,
        ),
        off_policy_trainer=True,
        trainer=dict(
            log_dir="./log/dqn_gym",
            writer=None,
            eps_collect=1.0,
            eps_collect_decay=0.6,
            eps_collect_min=0.01,
            eps_test=0.01,
        ),
        train=dict(
            epochs=200,
            iter_per_epoch=100,
            nstep_per_iter=10,
            learn_per_iter=1,
            test_per_epoch=10,
            max_reward=200,
            max_loss=10000,
        ),
    )

    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    train_dqn(cfg, TransformedDQN)
