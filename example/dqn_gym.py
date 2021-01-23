import gym

from rlcode.utils.train_dqn import TransformedDQN, train_dqn


def get_cfg() -> dict:
    cfg = dict(
        seed=42,
        env_fn=gym.make,
        env=dict(
            id="CartPole-v0",
        ),
        policy=dict(
            gamma=0.99,
            tau=0.9,
            target_update_freq=0,
            dist_log_freq=500,
            network=dict(
                layer_num=1,
                hidden_size=128,
                adv_layer_num=0,
                val_layer_num=0,
                activation=None,
            ),
            optim=dict(
                lr=1e-6,
            ),
        ),
        buffer=dict(
            buffer_size=10000,
            batch_size=128,
            alpha=0.0,
            beta=0.4,
        ),
        train_src=dict(
            nstep=128,
            max_episode_step=1000,
        ),
        test_src=dict(
            max_episode_step=1000,
        ),
        trainer=dict(
            writer="./log/dqn_gym",
            save_dir="./log/dqn_gym",
            eps_collect=0.6,
            eps_collect_decay=0.6,
            eps_collect_min=0.01,
            eps_test=0.01,
        ),
        train=dict(
            epochs=200,
            iter_per_epoch=1000,
            learn_per_iter=1,
            test_per_epoch=10,
            warmup_collect=4,
            max_reward=200,
        ),
    )

    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    train_dqn(cfg, TransformedDQN)
