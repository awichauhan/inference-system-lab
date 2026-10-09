from benchmark.metrics import percentile


def test_p50():

    values = [
        1,
        2,
        3,
        4,
        5,
    ]

    assert percentile(values, 50) == 3


def test_p95():

    values = list(range(1, 101))

    assert percentile(values, 95) == 95