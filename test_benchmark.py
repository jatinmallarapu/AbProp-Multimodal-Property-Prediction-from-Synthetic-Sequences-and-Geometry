import numpy as np
from benchmark import dataset, partition, metrics


def test_groups_never_cross_splits():
    _, _, y, groups = dataset(42)
    splits = partition(groups, 42)
    assert np.all(sum(s.astype(int) for s in splits) == 1)
    assert [len(np.unique(groups[s])) for s in splits] == [56, 12, 12]
    assert np.isfinite(y).all()


def test_identity_metrics():
    values = np.arange(12).reshape(4, 3)
    for scores in metrics(values, values).values():
        assert scores["rmse"] == 0
        assert np.isclose(scores["spearman"], 1)
