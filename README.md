# BearingShift

Leakage-free bearing fault diagnosis under operating-condition shift, on the Paderborn (KAt) bearing dataset.

**Question:** how well can a model classify bearing health (healthy / inner-race fault / outer-race fault) on *bearings it has never seen*, and how much do common evaluation shortcuts inflate the score?

> **Status: pipeline complete, real-data results not yet run.** Every number below is a `[placeholder]` until `scripts/reproduce.sh` has been run on the real dataset. Only numbers produced by scripts in this repo will be filled in.

## Why

Random window splits put windows from the *same recording* in both train and test. Recording- and condition-wise splits still put the *same physical bearing* on both sides. A model can then recognise the bearing instead of the fault. Recent work (Vieira et al., arXiv:2509.22267; Matania et al., PHME 2024, *"Test-Training Leakage in Evaluation of Machine Learning Algorithms for Condition-Based Maintenance"*) argues that this inflates reported accuracy. This project measures that inflation with the same features and models on every split.

## Method

1. **Tidy index:** one row per recording (bearing_id, condition, damage type, fault location, label, path). Labels come from `configs/paderborn_bearings.csv`, not from file names.
2. **Split first, window second.** 1 s non-overlapping windows. Everything learned (scaler, thresholds, models, hyperparameters) is fitted on training data only.
3. **Leakage ladder:**

   | Rung | Split | What leaks |
   |---|---|---|
   | a | random windows | same recording in train and test |
   | b | whole recordings | same bearing, same condition |
   | c | leave one condition out (4 folds) | same bearing, other condition |
   | d | bearing-wise, class-stratified, 20 repeats | nothing (no bearing on both sides) |

4. **Features (23 per window):**
   - time-domain: RMS, kurtosis, skewness, crest factor, peak-to-peak
   - 6 log band energies
   - envelope spectrum (band-pass 0.5–10 kHz → Hilbert envelope → FFT): peak-to-background ratio at BPFO, BPFI, BSF and FTF, harmonics 1–3, in **orders** of shaft speed

   Formulas (6203 bearing: 8 balls, d = 6.75 mm, D = 28.55 mm, φ = 0°):
   - BPFO = n/2·(1 − d/D·cosφ) = 3.054×
   - BPFI = n/2·(1 + d/D·cosφ) = 4.946×
   - BSF = D/(2d)·(1 − (d/D·cosφ)²) = 1.997×
   - FTF = ½·(1 − d/D·cosφ) = 0.382×

   Source: Randall & Antoni (2011); geometry from Lessmeier et al. (2016), Table 1.
5. **Models:** majority class, physics threshold rule, logistic regression on time-domain features, and LightGBM tuned by nested CV grouped like the outer split.
6. **Metrics:** macro-AUROC, macro-F1, per-class recall; window and recording level; mean ± std over splits.
7. **SHAP** on LightGBM (bearing-wise splits), with each feature's correlation to speed, torque and radial force.

Every choice and its reason is in [DECISIONS.md](DECISIONS.md).

## Results

Window level, LightGBM, mean ± std across splits (`results/leakage_ladder.csv`):

| Split | Macro-F1 | Macro-AUROC | n splits |
|---|---|---|---|
| a. random window | [placeholder] | [placeholder] | 20 |
| b. recording | [placeholder] | [placeholder] | 20 |
| c. condition | [placeholder] | [placeholder] | 4 |
| d. bearing (leakage-free) | [placeholder] | [placeholder] | 20 |

Baselines under the bearing-wise split: majority [placeholder], physics rule [placeholder], logistic regression [placeholder].

Plots: `results/leakage_ladder_window.png`, `results/leakage_ladder_recording.png`, `results/shap_importance.png` [to be generated].

SHAP finding (defect-frequency features vs condition artefacts): [placeholder].

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                      # 28 tests: split leakage, features on synthetic signals, metrics
bash scripts/reproduce.sh   # lists files + size, asks before downloading, then runs everything
```

Extraction needs `unrar`, `unar`, `7z` or `bsdtar`. To check the pipeline without the real data, `bash scripts/smoke_test.sh` runs everything on synthetic signals (outputs in `results/smoke/`, which are **not** results).

## Repository layout

```
bearingshift/   data.py splits.py windowing.py features.py models.py evaluate.py synthetic.py
configs/        default.yaml (real data), real_only.yaml, smoke.yaml (synthetic), paderborn_bearings.csv, paderborn_sha256.txt
scripts/        download_paderborn.py build_index.py run_experiment.py reproduce.sh smoke_test.sh
tests/          pytest suite
results/        CSVs and plots written by run_experiment.py
```

## Limitations

- Combined-damage bearings (KB23, KB24, KB27) are excluded, as in the dataset paper and Vieira et al.
- The main run mixes artificial and real damage; `configs/real_only.yaml` repeats it with healthy + real only.
- With only 6 healthy bearings, each bearing-wise test set holds 2 of them, so healthy-class recall is noisy.
- The envelope band is fixed rather than chosen per signal (e.g. kurtogram).
- BPFO lies 0.054 orders from the 3× shaft harmonic; at 1 s windows the first harmonic is close to the FFT resolution.
- Nominal rather than measured shaft speed.
- Vibration channel only; motor currents are unused.

## Next phases (not started)

Phase 2: CWRU, MiniRocket and a WDCNN-style 1D-CNN under bearing-wise splits, artificial→real transfer, ablation of order-normalised features, noise robustness. Phase 3: domain adaptation (CORAL/DANN) and a technical report.

## Data and citation

Paderborn University KAt bearing dataset, licence **CC BY-NC 4.0**. Cite:

> C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro. *Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification.* PHM Society European Conference, 2016.
