"""Guard reporting against empty windows and conditional denominators."""
import numpy as np
from nrtm.temporal.proportions import window_proportions, topic_trends


def test_no_eligible_windows_does_not_invent_a_trend():
    trends, excluded = topic_trends(np.array([[.2, .8]]), ['2015-2017'], [8])
    assert excluded == ['2015-2017']
    assert all(t['trend'] == 'unknown' and t['absolute_change'] is None for t in trends)


def test_outliers_are_excluded_but_total_count_remains_visible():
    proportions, counts = window_proportions([[1, 0], [0, 0], [0, 1]], ['a'], ['a'] * 3)
    np.testing.assert_allclose(proportions, [[.5, .5]])
    assert counts == [3]
