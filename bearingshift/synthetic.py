"""Synthetic Paderborn-like recordings, used ONLY for tests and pipeline smoke runs.

Nothing produced from this data is a result. It exists so that the full pipeline can be
checked end-to-end before (or without) the real dataset.

Physics of the fake signal: a bearing defect creates a short impact every time a ball
rolls over it. Each impact rings the structure at a resonance frequency, so the signal is
a train of decaying sine bursts repeating at the defect frequency (BPFO or BPFI).
"""
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import fftconvolve

from bearingshift.data import CONDITIONS


def impulse_train(fs, duration_s, rate_hz, rng, jitter=0.01):
    """Unit impulses at `rate_hz`, with small random timing jitter (real slip)."""
    n = int(fs * duration_s)
    x = np.zeros(n)
    period = 1.0 / rate_hz
    times = np.arange(0, duration_s, period)
    times = times + rng.normal(0, jitter * period, size=len(times))
    idx = np.clip((times * fs).astype(int), 0, n - 1)
    x[idx] = 1.0
    return x


def synth_recording(label, rpm, torque_nm, radial_force_n, bearing_seed, rec_seed,
                    fs=64000, duration_s=2.0, defect_orders=None):
    """One fake vibration recording for a bearing with the given label and condition."""
    rng_b = np.random.default_rng(bearing_seed)  # properties of the physical bearing
    rng = np.random.default_rng(rec_seed)        # recording-to-recording noise
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    shaft_hz = rpm / 60.0

    # Shaft harmonics (imbalance/misalignment), stronger with radial force.
    x = sum((radial_force_n / 1000) * 0.2 / h * np.sin(2 * np.pi * h * shaft_hz * t) for h in (1, 2, 3))
    # A bearing-specific tone: an "artefact" that lets leaky splits memorise bearings.
    x = x + 0.05 * np.sin(2 * np.pi * rng_b.uniform(10000, 15000) * t)

    if label in ("inner", "outer"):
        order = defect_orders["bpfi" if label == "inner" else "bpfo"]
        impulses = impulse_train(fs, duration_s, order * shaft_hz, rng)
        if label == "inner":
            # Inner-race defect rotates with the shaft, so impact strength varies once per rev.
            impulses *= 1 + 0.5 * np.cos(2 * np.pi * shaft_hz * t)
        resonance_hz = rng_b.uniform(3000, 7000)
        tk = np.arange(int(0.005 * fs)) / fs
        ring = np.exp(-tk / 0.0008) * np.sin(2 * np.pi * resonance_hz * tk)
        x = x + rng_b.uniform(0.5, 1.5) * fftconvolve(impulses, ring)[:n]

    noise_std = 0.2 * (0.5 + torque_nm)
    return x + rng.normal(0, noise_std, size=n)


def write_recording(path, x, channel="vibration_1"):
    """Save a signal in the same nested layout as a Paderborn .mat file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = np.zeros((1, 2), dtype=[("Name", "O"), ("Data", "O")])
    channels[0, 0] = ("speed", np.zeros(4))
    channels[0, 1] = (channel, x.astype(np.float32))
    savemat(path, {path.stem: {"Y": channels}})


def make_dataset(out_dir, bearing_table, defect_orders, n_repetitions=2, fs=64000, duration_s=2.0):
    """Write a full fake dataset: every labelled bearing x every condition x repetitions."""
    out_dir = Path(out_dir)
    labelled = bearing_table.dropna(subset=["label"]).reset_index(drop=True)
    for b, row in labelled.iterrows():
        for c, (cond, params) in enumerate(CONDITIONS.items()):
            for rep in range(1, n_repetitions + 1):
                x = synth_recording(row["label"], params["rpm"], params["torque_nm"], params["radial_force_n"],
                                    bearing_seed=1000 + b, rec_seed=b * 1000 + c * 100 + rep,
                                    fs=fs, duration_s=duration_s, defect_orders=defect_orders)
                bid = row["bearing_id"]
                write_recording(out_dir / bid / f"{cond}_{bid}_{rep}.mat", x)
