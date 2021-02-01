import gym

from rlcode.utils.train_pg import PG, train_pg


def get_cfg() -> dict:
    cfg = dict(
        seed=None,
        env_fn=gym.make,
        env=dict(
            id="CartPole-v0",
        ),
        policy=dict(
            dist_log_freq=500,
            device=None,
            gamma=0.99,
            batch_size=16,
            shuffle=True,
            network=dict(
                layer_num=2,
                hidden_size=1024,
                activation="ReLU",
            ),
            optim=dict(
                lr=1e-4,
            ),
        ),
        train_src=dict(
            max_episode_step=10000,
        ),
        test_src=dict(
            max_episode_step=None,
        ),
        trainer=dict(
            writer="./log/pg_gym",
            save_dir="./log/pg_gym",
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
            warmup_collect=0,
            max_reward=200,
            max_loss=10000,
        ),
    )

    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    train_pg(cfg, PG)
