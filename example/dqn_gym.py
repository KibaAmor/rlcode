import gym

from rlcode.utils.train_dqn import DQN, train_dqn


def get_cfg() -> dict:
    cfg = dict(
        seed=42,
        exp_per_collect=64,
        warmup_size=256,
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
                hidden_size=256,
                dueling=None,
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
        writer=dict(
            log_dir="./log/dqn_gym",
        ),
        train=dict(
            epochs=200,
            iter_per_epoch=1000,
            learn_per_iter=1,
            test_per_epoch=80,
            max_reward=200,
        ),
    )

    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    train_dqn(cfg, DQN)
