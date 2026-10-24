import numpy as np

from bearingshift.windowing import window_signal


def test_non_overlapping_windows_drop_the_tail():
    x = np.arange(10_500)
    w = window_signal(x, fs=1000, window_s=1.0)
    assert w.shape == (10, 1000)
    np.testing.assert_array_equal(w[1], np.arange(1000, 2000))  # contiguous, no overlap


def test_overlapping_windows():
    w = window_signal(np.arange(3000), fs=1000, window_s=1.0, overlap=0.5)
    assert w.shape == (5, 1000)
    assert w[1][0] == 500


def test_signal_shorter_than_window():
    assert window_signal(np.arange(10), fs=1000, window_s=1.0).shape == (0, 1000)
