# Decisions

Each choice, with why. "OPEN" marks things that still need checking.

## Data
- **Paderborn only for the MVP.** It has 32 physical bearings under 4 conditions, which makes bearing-wise and condition-wise splits possible. CWRU has very few bearings per class (phase 2).
- **Labels from a bearing table, not file names.** `configs/paderborn_bearings.csv` maps bearing → damage type → fault location. File names only give condition, bearing ID and repetition (the documented naming convention).
- **Table checked against the official documentation:** Tables 4, 5 and 7 of Lessmeier et al. (2016), the paper the KAt data-center page links for this mapping. All 32 bearings match, and each row cites its source table.
- **Combined-damage bearings (KB23, KB24, KB27) are excluded from the 3-class task.** The paper lists them as "IR (+OR)", "IR (+OR)" and "OR + IR", so a single label would be a guess. The dataset authors' own experiments (PHME16 Tables 8 and 10) don't use them, and neither does Vieira et al. (arXiv:2509.22267).
- **Artificial and real damage together in the main run; `configs/real_only.yaml` for healthy + real only.** Vieira et al. use only healthy + real bearings because artificial damage behaves differently. Running both lets us compare to them and see how much artificial bearings help or hurt.
- **Nominal shaft speed from the condition code (900 / 1500 rpm).** The official operating-conditions page says each parameter was held constant per measurement, so nominal speed is a fair approximation. Using the measured speed channel is a possible refinement.
- **Raw data stays out of git** (size, CC BY-NC licence). The download script lists sizes first and only downloads with `--yes`. SHA-256 checksums are recorded on first download and verified afterwards, because the site publishes none.

## Splits
- **Four rungs (random window → recording → condition → bearing).** Same features and models on every rung, so any gap between rungs is caused by the evaluation protocol alone.
- **Bearing-wise split: per class, 25% of bearings to test (at least 1), repeated 20 times.** Stratifying by class guarantees every test set contains all three classes, which macro-AUROC needs. Twenty repeats give a spread instead of one lucky split.
- **Condition-wise split has exactly 4 folds** (leave one condition out), so its std is over 4 values.
- **Inner tuning CV is grouped the same way as the outer split.** A project using a leaky protocol would also tune leakily, and the ladder should show the protocol as it is actually used.
- **Checked against Vieira et al. (arXiv:2509.22267).** Their PU leakage scenarios map onto our ladder: repetition-wise = our recording split, condition-wise = ours, plus segment-level leakage = our random-window split. They also use 20 bearing-wise splits, a class-stratified hold-out, and inner tuning splits kept separate from the evaluation splits, as here.
- **OPEN: the PHME 2024 paper (Matania et al., "Test-Training Leakage ...") was only seen as an abstract**, because papers.phmsociety.org is not on the network allowlist. Its stated point (incorrect train/test separation inflates performance) is what the ladder measures.

## Windowing
- **1 s, non-overlapping windows.** 1 s gives 1 Hz spectral resolution, enough to separate BPFO and BPFI. No overlap, because overlapping windows are near-duplicates and make the random-window leak even worse.
- **Split first, then window.** Recordings are split; windows are taken only from recordings already assigned to train or test. Window features depend only on the window itself (nothing is learned), so they are cached per recording. Everything that is learned (scaler, rule thresholds, models, hyperparameters) is fitted on training rows only. `tests/test_models.py` checks the scaler.

## Features
- **Three families: time-domain, band energies, envelope order peaks.** These are the classic vibration-diagnosis features, and the family label lets SHAP show what the model relies on.
- **Kurtosis is Pearson (3 for Gaussian noise).** Impacts from a defect push it above 3, which is easy to explain.
- **Band energies: 6 octave-like bands as log10 power from a Welch PSD.** The log compresses the large dynamic range.
- **Envelope band 500 Hz–10 kHz,** the band Vieira et al. use for PU after finding that most relevant content of the artificially damaged bearings lies there. A data-driven band (e.g. a kurtogram fitted on training data) is a later refinement.
- **Defect frequencies in orders** (frequency ÷ shaft frequency), so 900 and 1500 rpm align. Formulas and 6203 geometry (n=8, d=6.75 mm, D=28.55 mm, φ=0°, checked against PHME16 Table 1) are in `bearingshift/features.py`. This gives BPFO 3.054, BPFI 4.946, BSF 1.997, FTF 0.382 orders.
- **Peak-to-background ratio** (peak ÷ median envelope level over 0.5–12 orders) instead of raw amplitude. It is unitless, so it is less sensitive to load and sensor gain.
- **Harmonics 1–3, search ±1 FFT bin.** Known issue: BPFO (3.054×) is only 0.054 orders from the 3× shaft harmonic (≈1.35 Hz at 1500 rpm, close to the 1 Hz bin width). Harmonics 2 and 3 separate better (≈2.7 and ≈4 Hz), which is one reason to include them.

## Models
- **Baselines first:** majority class (the floor), a physics threshold rule (does textbook envelope analysis already solve it?), and logistic regression on time-domain features (how far simple statistics go).
- **Physics rule thresholds fitted on train by grid search over train quantiles, maximising macro-F1.** No test data touches the thresholds.
- **LightGBM** suits small tabular data with mixed-scale features and needs no scaling. `class_weight="balanced"` because healthy bearings are the minority (6 of 29).
- **Small grid (num_leaves, n_estimators, learning_rate), 3 inner folds.** Enough to show nested tuning without overfitting the inner CV on about 20 training bearings.

## Metrics
- **Macro-AUROC (one-vs-rest) and macro-F1.** Macro weights each class equally despite imbalance. AUROC measures ranking, F1 measures decisions at argmax.
- **Per-class recall,** because the cost of missing a fault differs from a false alarm.
- **Window and recording level.** Recording level averages window probabilities, closer to how a monitoring system decides.
- **Mean ± std across splits, never one split.**
- **Leakage alarm:** bearing-wise LightGBM macro-F1 > 0.95 prints a warning. Too-good leakage-free scores are treated as a bug until investigated.

## Interpretability
- **SHAP (TreeExplainer) on the bearing-wise splits only,** because that is the honest setting. Mean |SHAP| is averaged over test windows and classes, then over splits.
- **Spearman correlation of each feature with speed, torque and radial force.** An important feature that mostly tracks the operating condition is a red flag for a condition shortcut.
