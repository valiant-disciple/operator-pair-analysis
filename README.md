# Multimodal Sensing of Operator Pairs

Analysis code accompanying the BTech thesis **_Multimodal Sensing of Operator
Pairs: A Dual Eye-Tracking Pipeline for Coordination-Failure Prediction_**
(Arya Sikder, Dept. of Chemical Engineering, IIT Madras; guide: Prof.
Rajagopalan Srinivasan). The work is a methodological precursor to operator-pair
monitoring for chemical-engineering control rooms, evaluated on a laboratory
analogue of the board–field coordination problem (the HCRC Map Task).

## Companion repository

The experiment user interface (React/TypeScript Director–Matcher application,
Express WebSocket backend, WearOS heart-rate streaming app) lives in a separate
repository:

> **UI + data-collection stack:** https://github.com/valiant-disciple/map-task

This repository contains the **analysis** code. The two together cover the
full apparatus → features → model → results pipeline reported in the thesis.

---

## Repository layout

```
pipeline/    Raw recording -> per-trial feature streams
analysis/    Master CSV assembly and per-trial annotation
modeling/    Predictor, clustering, statistical-null evaluation
figures/     Figure-generation scripts for the thesis and the paper
thesis/      LaTeX source of the BTech thesis (compilable with pdflatex)
```

### `pipeline/` — recording to features

| File | What it does | Thesis reference |
|---|---|---|
| `01_preprocess_eye.py`         | Parses Aurora ET (iMotions CSV) and SmartEye Pro 10 (.log TSV) eye-tracker exports; projects gaze into the shared $651 \times 900$ map-space frame. | §2 Apparatus |
| `02_postprocess_features.py`   | Cardiac feature pipeline: HR mean, RMSSD, LF/HF, sample entropy, DFA $\alpha_1$. Joint cardiac coupling: cross-CRQA (DET, LAM), MdRQA, windowed cross-correlation, transfer entropy. | §2.4, Appendix B |
| `03_gaze_features.py`          | Per-operator gaze descriptors (fixation duration, scan length, dispersion, SGE, pupil). Joint gaze coupling: cross-CRQA, AOI overlap, gaze convergence within 100 px, leader/follower lag. Cross-modal: Director gaze × Matcher HR. | §2.4, §3.2 |
| `04_drawing_features.py`       | Stroke-level drawing kinematics. Outcome-dependent features (Chamfer, IoU, route coverage) are excluded from the predictor on endogeneity grounds (Kapoor 2023). | §2.4 |
| `05_llm_dialogue_annotation.py`| `gpt-5-mini` structured-output annotation of per-trial transcripts: route plan, repair sequences, common-ground events, misalignment events, errors, dropouts. | §2.3 |
| `06_lsl_flash_sender.py`<br>`07_lsl_flash_receiver.py` | LSL flash-sync between the Director browser clock and the SmartEye PC clock (validated 48,591,233 ms offset, median residual −10 ms across 37 sessions). | §2 Apparatus |

### `analysis/`

| File | What it does | Thesis reference |
|---|---|---|
| `build_master.py`         | Assembles per-trial master CSV across all 40 dyads × 225 trials, merging cardiac, gaze, audio, survey, and LLM-derived features. | §2.4 |
| `build_map_difficulty.py` | Per-map difficulty stratification (chamfer-based qcut tertiles: EASY / MED / HARD; per-map empirical target-reach rate). | §3.5.2 |

### `modeling/`

| File | What it does | Thesis reference |
|---|---|---|
| `multi_model_benchmark.py`     | Cross-pair predictor benchmark across feature pools (HR, gaze, speech, joint coupling, D+M, D+M+J). 10-fold GroupKFold × 3 seeds, in-fold mRMR + correlation pruning. Soft-vote ensemble: HistGB + LightGBM + XGBoost, Optuna-tuned. | §3.1, Table 3.1 |
| `passive_multimodal_v2.py`     | Passive (non-outcome-dependent) multimodal predictor used for the headline AUC of 0.761 ± 0.010. | §3.1 |
| `non_endogenous_predictor.py`  | Strict no-drawing variant; verifies the headline AUC is not driven by drawing-extent functionals. | §2.4, §3.1 |
| `failure_mode_clustering.py`   | K-means at k = 4 on the principal components of a 15-feature within-pair-z-scored physiology fingerprint. Cluster signature assignment (Director-Overloaded, Matcher-Disengaged, Director-Disengaged, Calm-Decoupled). | §3.4, Appendix A |
| `feature_importance.py`        | SHAP feature importance on the multimodal predictor, with per-modality colour-coded ranking. | §3.5.1 |
| `permutation_null.py`          | 200-iteration within-fold within-pair label-shuffle null. Empirical p < 0.005. | Appendix A |

### `figures/`

| File | What it produces (thesis figures) |
|---|---|
| `build_live_trial_slides.py`   | `joint_workload`, `cluster_naming_logic`, `k_choice_interpretability`, `random_trial_examples`, `behavioral_signature`, `cluster_effect_sizes`, `cluster_stability`, `cluster_signature_heatmap`, plus per-cluster exemplar traces. The single script that drives most of the appendix figures and the validation deck. |
| `build_paper_figures.py`       | `fig1_multimodal` (headline AUC + per-operator contributions) and `fig2_modes` (four physiology-derived failure modes, role asymmetry, per-difficulty dissociation). |

### `thesis/`

LaTeX source of the report. Compile with `pdflatex thesis.tex && bibtex thesis
&& pdflatex thesis.tex && pdflatex thesis.tex` from inside `thesis/`. Final PDF
is included as `thesis.pdf`.

---

## Reproducing the headline result

The pipeline assumes per-trial recordings have been captured by the UI in
[`valiant-disciple/map-task`](https://github.com/valiant-disciple/map-task).

```bash
# 1. Pre-process the eye-tracker exports and the cardiac stream
python pipeline/01_preprocess_eye.py --format aurora    --eye-file <Director.csv> ...
python pipeline/01_preprocess_eye.py --format smarteye  --eye-file <Matcher.log> ...
python pipeline/02_postprocess_features.py  --session <session_dir>

# 2. Build per-trial gaze and drawing feature streams
python pipeline/03_gaze_features.py    --session <session_dir>
python pipeline/04_drawing_features.py --session <session_dir>

# 3. LLM dialogue annotation (one structured-output call per trial)
python pipeline/05_llm_dialogue_annotation.py --transcripts <whisper_dir>

# 4. Assemble the master CSV across all dyads
python analysis/build_master.py --batch-out <batch_out_dir>
python analysis/build_map_difficulty.py

# 5. Run the predictor benchmark and the failure-mode clustering
python modeling/multi_model_benchmark.py
python modeling/passive_multimodal_v2.py
python modeling/failure_mode_clustering.py
python modeling/feature_importance.py
python modeling/permutation_null.py

# 6. Regenerate the thesis figures
python figures/build_paper_figures.py
python figures/build_live_trial_slides.py
```

---

## Data

The raw participant data (eye-tracker exports, audio, smartwatch HR streams,
survey responses) are **not** included in this repository. They are held under
the original study ethics protocol and can be requested from the authors.

The reproduction commands above assume the
`batch_out/master_with_speech_llm.csv` and `batch_out/failure_modes.csv`
files have been produced from the raw data. The headline results
(`headline_seed_pooled.csv`, `permutation_null_distribution.csv`, etc.) are
deterministic given the master CSVs and the random seeds in each modeling
script.

---

## Headline numbers (Table 3.1 of the thesis)

| Feature pool                                            | AUC                |
|---------------------------------------------------------|--------------------|
| HR only                                                 | 0.457              |
| Gaze only                                               | 0.598              |
| Speech only (best single)                               | 0.654              |
| Joint coupling only                                     | 0.551              |
| D+M (per-operator marginals)                            | 0.728              |
| **D+M+J multimodal (tuned soft-vote ensemble)**         | **0.761 ± 0.010**  |

10-fold GroupKFold (dyad as group) × 3 random seeds; in-fold mRMR top-40 plus
correlation pruning at $|r| < 0.85$. Survives 200-permutation within-pair
label-shuffle null at $p < 0.005$.

---

## Failure-mode taxonomy (Table 3.2 / Appendix A)

K-means at k = 4 on within-pair-z-scored physiology fingerprint (15 features,
79 trials with full bilateral coverage). Cluster stability across 50 random
seeds: within-cluster co-assignment 0.77 vs. between-cluster 0.11.

| Mode                  |  n | Fail rate | Wilson 95 % CI |
|-----------------------|----|-----------|----------------|
| Director-Overloaded   | 21 |    48 %   | [28 %, 68 %]   |
| Matcher-Disengaged    | 23 |    57 %   | [37 %, 74 %]   |
| Director-Disengaged   | 30 |    53 %   | [36 %, 70 %]   |
| Calm-Decoupled        |  5 |     0 %   | [ 0 %, 43 %]   |

---

## Citation

If you use this work, please cite the thesis:

```bibtex
@phdthesis{sikder2026multimodal,
  author = {Sikder, Arya},
  title  = {Multimodal Sensing of Operator Pairs: A Dual Eye-Tracking Pipeline
            for Coordination-Failure Prediction},
  school = {Indian Institute of Technology Madras},
  type   = {{B.Tech.}\ thesis},
  year   = 2026,
  note   = {Guide: Prof.\ Rajagopalan Srinivasan, Dept.\ of Chemical Engineering}
}
```

---

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Raw participant data is held separately under the study's ethics protocol and
is not covered by this licence.

---

## Acknowledgements

I thank my research guide Prof. Rajagopalan Srinivasan for his guidance, and
the 47 participant pairs and lab staff who supported the six-modality
recording protocol. The analysis benefited from open-source contributions to
scikit-learn, LightGBM, XGBoost, PyRQA, OpenAI Whisper, and matplotlib.
