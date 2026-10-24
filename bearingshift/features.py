"""Hand-crafted features for one window of vibration signal.

Three families:
  1. time-domain statistics: how loud and how "spiky" the signal is
  2. band energies: where in frequency the energy sits
  3. envelope-spectrum peaks at the bearing defect frequencies, expressed in ORDERS
     (multiples of shaft speed) so 900 rpm and 1500 rpm recordings are comparable

Defect frequencies (standard rolling-bearing kinematics, e.g. Randall & Antoni 2011,
"Rolling element bearing diagnostics - A tutorial", MSSP 25(2)), as multiples of shaft
frequency f_r, with n balls, ball diameter d, pitch diameter D, contact angle phi:
  BPFO = n/2 * (1 - d/D cos phi)          outer-race defect
  BPFI = n/2 * (1 + d/D cos phi)          inner-race defect
  BSF  = D/(2d) * (1 - (d/D cos phi)^2)   ball (rolling element) spin
  FTF  = 1/2 * (1 - d/D cos phi)          cage
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

FAMILY_PREFIXES = {"time": ("rms", "kurtosis", "skewness", "crest_factor", "peak_to_peak"),
                   "band": ("band_",), "envelope": ("env_",)}
TIME_FEATURES = ["rms", "kurtosis", "skewness", "crest_factor", "peak_to_peak"]


@dataclass(frozen=True)
class BearingGeometry:
    n_balls: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0


def defect_orders(g):
    """Defect frequencies in orders (divide by shaft frequency in Hz)."""
    ratio = g.ball_diameter_mm / g.pitch_diameter_mm * np.cos(np.radians(g.contact_angle_deg))
    return {
        "bpfo": g.n_balls / 2 * (1 - ratio),
        "bpfi": g.n_balls / 2 * (1 + ratio),
        "bsf": g.pitch_diameter_mm / (2 * g.ball_diameter_mm) * (1 - ratio**2),
        "ftf": 0.5 * (1 - ratio),
    }


def feature_family(name):
    for family, prefixes in FAMILY_PREFIXES.items():
        if name.startswith(prefixes):
            return family
    raise ValueError(f"Unknown feature family for {name}")


def time_domain_features(x):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2))
    return {
        "rms": rms,
        "kurtosis": stats.kurtosis(x, fisher=False),  # Pearson: 3 for Gaussian noise, >3 when impulsive
        "skewness": stats.skew(x),
        "crest_factor": np.max(np.abs(x)) / rms,
        "peak_to_peak": np.ptp(x),
    }


def band_energies(x, fs, bands):
    """log10 of signal power in each [low, high) Hz band, from a Welch PSD."""
    f, pxx = signal.welch(x, fs=fs, nperseg=min(4096, len(x)))
    out = {}
    for low, high in bands:
        mask = (f >= low) & (f < high)
        out[f"band_{int(low)}_{int(high)}"] = np.log10(np.trapezoid(pxx[mask], f[mask]) + 1e-20)
    return out


def amplitude_spectrum(x, fs):
    """Single-sided amplitude spectrum with a Hann window. A sine of amplitude A peaks at ~A."""
    window = np.hanning(len(x))
    amp = 2 * np.abs(np.fft.rfft(x * window)) / window.sum()
    return np.fft.rfftfreq(len(x), d=1 / fs), amp


def envelope_spectrum(x, fs, band):
    """Band-pass -> Hilbert envelope -> amplitude spectrum of the envelope."""
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, x)
    envelope = np.abs(signal.hilbert(filtered))
    return amplitude_spectrum(envelope - envelope.mean(), fs)


def envelope_order_features(x, fs, shaft_hz, band, orders, n_harmonics, search_bins, background_orders):
    """Envelope-spectrum peak at each defect order and harmonic, divided by the median
    envelope level (a unitless peak-to-background ratio, robust to sensor gain and load)."""
    f, amp = envelope_spectrum(x, fs, band)
    df = f[1] - f[0]
    in_background = (f / shaft_hz >= background_orders[0]) & (f / shaft_hz <= background_orders[1])
    background = np.median(amp[in_background]) + 1e-20

    out = {}
    for name, order in orders.items():
        for h in range(1, n_harmonics + 1):
            centre = int(round(order * h * shaft_hz / df))
            lo, hi = max(centre - search_bins, 0), centre + search_bins + 1
            out[f"env_{name}_h{h}"] = amp[lo:hi].max() / background
    return out


def window_features(x, fs, shaft_hz, orders, fcfg):
    """All features for one window. `fcfg` is the config's `features` section."""
    return {
        **time_domain_features(x),
        **band_energies(x, fs, fcfg["bands_hz"]),
        **envelope_order_features(x, fs, shaft_hz, fcfg["envelope_band_hz"], orders, fcfg["n_harmonics"],
                                  fcfg["peak_search_bins"], fcfg["background_orders"]),
    }
