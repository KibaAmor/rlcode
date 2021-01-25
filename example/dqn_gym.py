import gym

from rlcode.utils.train_dqn import DQN, train_dqn


def get_cfg() -> dict:
    cfg = dict(
        seed=None,
        env_fn=gym.make,
        env=dict(
            id="CartPole-v0",
        ),
        policy=dict(
            gamma=0.99,
            tau=1.0,
            target_update_freq=500,
            dist_log_freq=500,
            network=dict(
                layer_num=3,
                hidden_size=256,
                adv_layer_num=2,
                val_layer_num=2,
                activation="ReLU",
            ),
            optim=dict(
                lr=1e-4,
            ),
        ),
        buffer=dict(
            buffer_size=10000,
            batch_size=128,
            alpha=0.0,
            beta=0.4,
        ),
        train_src=dict(
            nstep=10,
            max_episode_step=1000,
        ),
        test_src=dict(
            max_episode_step=None,
        ),
        trainer=dict(
            writer="./log/dqn_gym",
            save_dir="./log/dqn_gym",
            eps_collect=1.0,
            eps_collect_decay=0.6,
            eps_collect_min=0.01,
            eps_test=0.01,
        ),
        train=dict(
            epochs=200,
            iter_per_epoch=1000,
            learn_per_iter=1,
            test_per_epoch=10,
            warmup_collect=60,
            max_reward=200,
            max_loss=10000,
        ),
    )

    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    train_dqn(cfg, DQN)
