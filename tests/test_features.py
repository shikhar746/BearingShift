import numpy as np
import pytest

from bearingshift.features import (BearingGeometry, amplitude_spectrum, band_energies, defect_orders,
                                   envelope_order_features, envelope_spectrum, time_domain_features)
from bearingshift.synthetic import synth_recording

FS = 64000
T = np.arange(FS) / FS  # 1 s
GEOMETRY_6203 = BearingGeometry(n_balls=8, ball_diameter_mm=6.75, pitch_diameter_mm=28.55)


def test_157hz_sine_gives_157hz_peak():
    f, amp = amplitude_spectrum(np.sin(2 * np.pi * 157 * T), FS)
    assert f[np.argmax(amp)] == pytest.approx(157, abs=1)
    assert amp.max() == pytest.approx(1.0, rel=0.05)


def test_envelope_recovers_modulation_frequency():
    # A 5 kHz carrier whose amplitude pulses 157 times a second: the envelope is the pulse.
    x = (1 + np.cos(2 * np.pi * 157 * T)) * np.sin(2 * np.pi * 5000 * T)
    f, amp = envelope_spectrum(x, FS, band=[3000, 7000])
    assert f[np.argmax(amp)] == pytest.approx(157, abs=1)


def test_time_domain_features_of_a_sine():
    feats = time_domain_features(3 * np.sin(2 * np.pi * 50 * T))
    assert feats["rms"] == pytest.approx(3 / np.sqrt(2), rel=1e-3)
    assert feats["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert feats["kurtosis"] == pytest.approx(1.5, rel=1e-2)
    assert feats["peak_to_peak"] == pytest.approx(6, rel=1e-3)


def test_band_energy_lands_in_the_right_band():
    feats = band_energies(np.sin(2 * np.pi * 3000 * T), FS, [[0, 2000], [2000, 4000], [4000, 8000]])
    assert max(feats, key=feats.get) == "band_2000_4000"


def test_defect_orders_for_6203():
    orders = defect_orders(GEOMETRY_6203)
    ratio = 6.75 / 28.55
    assert orders["bpfo"] == pytest.approx(4 * (1 - ratio))
    assert orders["bpfi"] == pytest.approx(4 * (1 + ratio))
    assert orders["bpfo"] + orders["bpfi"] == pytest.approx(8)  # n_balls, by construction
    assert orders["ftf"] == pytest.approx(orders["bpfo"] / 8)


@pytest.mark.parametrize("label, strong, weak", [("outer", "bpfo", "bpfi"), ("inner", "bpfi", "bpfo")])
def test_envelope_features_point_at_the_faulty_race(label, strong, weak):
    orders = defect_orders(GEOMETRY_6203)
    x = synth_recording(label, 1500, 0.7, 1000, bearing_seed=1, rec_seed=2, duration_s=1.0, defect_orders=orders)
    feats = envelope_order_features(x, FS, shaft_hz=25.0, band=[2000, 10000], orders=orders,
                                    n_harmonics=3, search_bins=1, background_orders=[0.5, 12.0])
    assert feats[f"env_{strong}_h1"] > 3 * feats[f"env_{weak}_h1"]
