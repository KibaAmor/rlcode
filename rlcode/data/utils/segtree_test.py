from rlcode.data.utils.segtree import SumSegmentTree


def test_SumSegmentTree():
    data = [3.0, 9.0, 2.0, 0.0, 6.0, 7.0, 1.0, 8.0, 5.0, 4.0]

    st = SumSegmentTree(len(data))
    for i, v in enumerate(data):
        st[i] = v

    assert len(data) == len(st)
    assert all([a == b for a, b in zip(data, st)])
    assert st.reduce() == sum(st)

    for left in range(len(st)):
        for right in range(left + 1, len(st)):
            assert st.reduce(left, right) == sum(data[left:right])

    def get_prefix_sum_index(st, value):
        for i in range(len(st)):
            if st[i] >= value:
                return i
            value -= st[i]
        assert False

    for v in range(int(sum(st))):
        assert get_prefix_sum_index(st, v) == st.get_prefix_sum_index(v)


if __name__ == "__main__":
    test_SumSegmentTree()
