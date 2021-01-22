from rlcode.data.buffer import PrioritizedReplayBuffer, ReplayBuffer


def base_test(buffer: ReplayBuffer):
    buffer_size = buffer.buffer_size
    batch_size = buffer.batch_size

    for i in range(buffer_size):
        assert i == len(buffer)
        buffer.add(1, 2, 3, False, 4)

    assert buffer_size == len(buffer)
    for i in range(buffer_size):
        assert buffer_size == len(buffer)
        buffer.add(1, 2, 3, False, 4)

    batch = buffer.sample()
    assert (
        batch_size
        == len(batch.obss)
        == len(batch.acts)
        == len(batch.rews)
        == len(batch.dones)
        == len(batch.next_obss)
        == len(batch.indexes)
    )
    if batch.weights is not None:
        assert batch_size == len(batch.weights)


def test_ReplayBuffer():
    base_test(ReplayBuffer(10, 2))


def test_PrioritizedReplayBuffer():
    base_test(PrioritizedReplayBuffer(10, 2, 0.6, 0.4))


if __name__ == "__main__":
    test_ReplayBuffer()
    test_PrioritizedReplayBuffer()
