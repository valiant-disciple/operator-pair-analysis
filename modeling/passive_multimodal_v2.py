#!/usr/bin/env python3
"""
Passive Multimodal v2 — runs after Phase 2B + 3A complete.

Tests whether cutting-edge features (coherence, Granger, MI, entropy, W_team, windowed)
push passive AUC above 0.85.

Outcomes:
  - target_reached (binary, AUC) — primary HCI
  - chamfer_within_dyad_z (continuous, R²/r) — within-dyad effort
  - composite_accuracy_pca1 (continuous)

Models compared:
  - Baseline passive (Phase 1A)
  - Passive + 1B (sophisticated cross-modal)
  - Passive + 1B + 2B (coherence/Granger/MI/entropy)
  - Passive + 1B + 2B + W_team (paper headline composite)
  - Passive + 1B + 2B + W_team + windowed
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

BATCH_OUT = Path("/Users/kolosus/Documents/DSA/analysis/batch_out")
MASTER = BATCH_OUT / "master_clean.csv"


def to_bool(s):
    if pd.isna(s): return float('nan')
    if isinstance(s, bool): return float(s)
    s = str(s).strip().lower()
    if s in ('true','1','yes'): return 1.0
    if s in ('false','0','no'): return 0.0
    return float('nan')


EXCLUDE_BASE = {'dyad_id','session_dir','trial','trial_position','sessionId','sessionId_director','sessionId_matcher',
    'mapNumber','trial_start_ms','trial_start_iso','trial_end_ms','trial_end_iso','director_note','ref_clock',
    'audio_count','clock_offset_ms','target_reached_num','demo_gender_pair','demo_gender_d','demo_gender_m',
    'target', 'target_reached'}
OUTCOMES_BASE = {'map_iou','map_f1','map_chamfer','map_hausdorff','map_dice','map_ssim','map_boundary_f1',
    'map_boundary_p','map_boundary_r','map_precision','map_recall','coverage_gt','coverage_pred',
    'target_reached','path_confidence','chamfer_map_adjusted','iou_map_adjusted','boundary_f1_map_adjusted',
    'chamfer_within_dyad_rank','chamfer_within_dyad_z','composite_accuracy_pca1','composite_accuracy_pca2',
    'map_difficulty_chamfer','map_difficulty_iou','map_difficulty_boundary_f1','map_difficulty_target_rate'}


def is_subjective(c):
    return c.startswith('tlx_') or (c.startswith('psmm_') and not c.startswith('psmm_dir_mat')) or c.startswith('demo_')


def is_drawing(c):
    return c.startswith('draw_') or c in ('strokes', 'strokePoints')


def is_phase1b(c):
    return any(c.startswith(p) for p in [
        'psmm_dir_mat', 'pupil_hr_', 'crossmodal_hr_pupil', 'gaze_lead_stroke_',
        'arousal_d_', 'arousal_m_', 'hr_sync_', 'gaze_within_trial', 'hr_stroke_pre_change',
        'speech_x_gaze', 'gaze_xy_xcorr', 'speech_to_stroke', 'late_window_'
    ]) or c in ('hr_phase_plv', 'hr_mean_phase_diff')


def is_phase2b(c):
    return any(c.startswith(p) for p in [
        'coh_', 'granger_', 'mi_', 'perm_ent_', 'joint_state_ent_', 'convergence_',
        'hr_gaze_entropy_', 'gaze_entropy_dyad'
    ])


def is_wteam(c):
    return c.startswith('wteam_')


def is_windowed(c):
    return c.startswith('pre_')


def is_passive_baseline(c):
    """Phase 1A passive: HR, gaze, prosody (no Phase 1B/2B/Wteam/windowed)."""
    return ((c.startswith('hr_') or c.startswith('gaze_') or c.startswith('prosody_'))
            and not is_phase1b(c) and not is_phase2b(c) and not is_wteam(c) and not is_windowed(c))


def get_features(df, model_name):
    cols = [c for c in df.columns if c not in EXCLUDE_BASE and c not in OUTCOMES_BASE]
    if model_name == 'M0_null':
        return []
    elif model_name == 'TLX-only':
        return [c for c in cols if c.startswith('tlx_')]
    elif model_name == 'PSMM-only':
        return [c for c in cols if c.startswith('psmm_') and not c.startswith('psmm_dir_mat')]
    elif model_name == 'Subjective (TLX+PSMM)':
        return [c for c in cols if is_subjective(c)]
    elif model_name == 'Drawing-only':
        return [c for c in cols if is_drawing(c)]
    elif model_name == 'Passive baseline (1A)':
        return [c for c in cols if is_passive_baseline(c)]
    elif model_name == 'Passive + 1B':
        return [c for c in cols if is_passive_baseline(c) or is_phase1b(c)]
    elif model_name == 'Passive + 1B + 2B':
        return [c for c in cols if is_passive_baseline(c) or is_phase1b(c) or is_phase2b(c)]
    elif model_name == 'Passive + 1B + 2B + Wteam':
        return [c for c in cols if is_passive_baseline(c) or is_phase1b(c) or is_phase2b(c) or is_wteam(c)]
    elif model_name == 'Passive ALL (incl. windowed)':
        return [c for c in cols if is_passive_baseline(c) or is_phase1b(c) or is_phase2b(c) or is_wteam(c) or is_windowed(c)]
    elif model_name == 'Wteam-only':
        return [c for c in cols if is_wteam(c)]
    elif model_name == 'Windowed-only':
        return [c for c in cols if is_windowed(c)]
    elif model_name == 'Coherence-only (2B)':
        return [c for c in cols if is_phase2b(c)]
    elif model_name == 'FULL no drawing':
        return [c for c in cols if not is_drawing(c)]
    elif model_name == 'FULL all':
        return cols
    return []


def evaluate_classification(df, fcols, y, groups):
    valid = ~np.isnan(y)
    if valid.sum() < 30: return float('nan'), float('nan')
    fcols = [c for c in fcols if c in df.columns]
    if not fcols and len(set(groups[valid])) >= 5:
        return 0.5, float(y[valid].mean())  # null
    if not fcols: return float('nan'), float('nan')
    X = df[fcols].values[valid]
    yv = y[valid]; gv = groups[valid]
    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(yv, dtype=float)
    for tr_idx, te_idx in logo.split(X, yv, gv):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X[tr_idx])
        Xte = imp.transform(X[te_idx])
        m = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)
        m.fit(Xtr, yv[tr_idx])
        y_pred[te_idx] = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yv, y_pred) if len(set(yv)) > 1 else float('nan')
    ap = average_precision_score(yv, y_pred) if len(set(yv)) > 1 else float('nan')
    return auc, ap


def evaluate_regression(df, fcols, y, groups):
    valid = ~np.isnan(y)
    if valid.sum() < 30: return float('nan'), float('nan')
    fcols = [c for c in fcols if c in df.columns]
    if not fcols: return -1.0, 0.0
    X = df[fcols].values[valid]
    yv = y[valid]; gv = groups[valid]
    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(yv, dtype=float)
    for tr_idx, te_idx in logo.split(X, yv, gv):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X[tr_idx])
        Xte = imp.transform(X[te_idx])
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)
        m.fit(Xtr, yv[tr_idx])
        y_pred[te_idx] = m.predict(Xte)
    return r2_score(yv, y_pred), float(pearsonr(yv, y_pred)[0])


def main():
    df = pd.read_csv(MASTER)
    for c in df.columns:
        if c not in EXCLUDE_BASE:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['target'] = df['target_reached'].apply(to_bool)
    y_target = df['target'].values
    y_chamfer_z = df['chamfer_within_dyad_z'].values if 'chamfer_within_dyad_z' in df.columns else df['map_chamfer'].values
    y_pca1 = df['composite_accuracy_pca1'].values if 'composite_accuracy_pca1' in df.columns else None
    groups = df['dyad_id'].values

    models = ['M0_null', 'TLX-only', 'PSMM-only', 'Subjective (TLX+PSMM)', 'Drawing-only',
              'Passive baseline (1A)', 'Passive + 1B', 'Passive + 1B + 2B', 'Passive + 1B + 2B + Wteam',
              'Passive ALL (incl. windowed)', 'Wteam-only', 'Windowed-only', 'Coherence-only (2B)',
              'FULL no drawing', 'FULL all']

    print(f"{'Model':<35} {'n_feat':>7} {'AUC_target':>10} {'AP':>8} {'R²_chmfz':>10} {'r_chmfz':>9} {'R²_pca1':>9}")
    print('-' * 100)
    rows = []
    for model in models:
        feats = get_features(df, model)
        n = len(feats)
        auc, ap = evaluate_classification(df, feats, y_target, groups)
        r2_c, r_c = evaluate_regression(df, feats, y_chamfer_z, groups)
        if y_pca1 is not None:
            r2_p, r_p = evaluate_regression(df, feats, y_pca1, groups)
        else:
            r2_p, r_p = float('nan'), float('nan')
        print(f"{model:<35} {n:>7} {auc:>10.3f} {ap:>8.3f} {r2_c:>+10.4f} {r_c:>+9.3f} {r2_p:>+9.4f}")
        rows.append({
            'model': model, 'n_features': n,
            'auc_target': round(auc, 4) if not math.isnan(auc) else '',
            'ap_target': round(ap, 4) if not math.isnan(ap) else '',
            'r2_chamfer_z': round(r2_c, 4) if not math.isnan(r2_c) else '',
            'r_chamfer_z': round(r_c, 4) if not math.isnan(r_c) else '',
            'r2_pca1': round(r2_p, 4) if not math.isnan(r2_p) else '',
        })

    pd.DataFrame(rows).to_csv(BATCH_OUT / 'passive_v2_results.csv', index=False)
    print(f"\nWrote {BATCH_OUT / 'passive_v2_results.csv'}")


if __name__ == '__main__':
    main()
