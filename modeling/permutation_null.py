#!/usr/bin/env python3
"""Phase 50: Honest cross-validation pipeline.

Addresses three real issues found in the previous pipelines:
  1. Global feature selection on full data (phase37, fix02b) — feature-selection leakage.
  2. Hardcoded single-modality AUCs in phase17_figures.py — never computed on our data.
  3. W composites constructed using outcome labels (phase28) — in-sample features fed back.

This script runs:
  A. Headline multimodal stacking ensemble
       5-fold GroupKFold × 5 seeds × in-fold mRMR top-K × HistGB+LightGBM+XGBoost+LR-L1+RF stacking.
  B. Single-modality stacking pipelines (HR-only, gaze-only, speech-only, drawing-only).
  C. Modality ablation re-run with same CV protocol (single HistGB for speed).
  D. Within-fold permutation null (HistGB, 200 permutations, 5-fold GroupKFold).

Output: batch_out/phase50/
  - headline_5fold_gkf.csv
  - single_modality_5fold_gkf.csv
  - ablation_5fold_gkf.csv
  - permutation_null_5fold_gkf.csv
  - top_features_per_fold.csv
"""
import warnings
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                                StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

DSA = Path('/Users/kolosus/Documents/DSA')
BATCH = DSA / 'analysis' / 'batch_out'
TRIAL = BATCH / 'master_with_speech_llm.csv'
OUT_DIR = BATCH / 'phase50'
OUT_DIR.mkdir(exist_ok=True)


def to_bool(s):
    if pd.isna(s):
        return float('nan')
    s = str(s).strip().lower()
    if s in ('true', '1', 'yes'):
        return 1.0
    if s in ('false', '0', 'no'):
        return 0.0
    return float('nan')


# Same leak filter as fix02b
def is_outcome_leak(c):
    cl = c.lower()
    if cl.startswith('draw_ds_'):
        return True
    if any(k in cl for k in ['target_reached', 'reached_', 'success_label', 'failure_label']):
        return True
    LEAKS = ['chamfer', 'hausdorff', 'composite_accuracy', 'landmark_coverage',
             'route_coverage', 'path_confidence', 'coverage_pred', 'iou_',
             'pca_recon_err', 'spatial_accuracy', 'completeness',
             'landmarks_covered', 'success_rate', 'target_rate',
             'map_difficulty_target', 'fail_rate', 'pct_success']
    if any(k in cl for k in LEAKS):
        return True
    if cl in ('chamfer', 'iou', 'recall', 'precision', 'f1'):
        return True
    # Also exclude outcome-constructed W composites for honest evaluation
    if cl in ('w_director', 'w_matcher', 'w_team'):
        return True
    return False


# Modality bucketing by column-name prefix
HR_PREFIXES = ('hr_', 'hrv_')
HR_KEYS_IN = ('perm_ent_hr', 'crossmodal_hr_', 'coh_hr', 'dtw_hr', 'convergence_hr', 'hr_sync')
GAZE_PREFIXES = ('gaze_', 'pup', 'wav_')
GAZE_KEYS_IN = ('pre_early_pupil', 'pre_mid_pupil', 'pre_late_pupil', 'pre_whole_pupil',
                'perm_ent_pupil', 'pupil')
SPEECH_PREFIXES = ('spch_', 'lex_', 'dlg_', 'prosody_', 'speech_')
SPEECH_KEYS_IN = ('audio_count',)
DRAW_PREFIXES = ('draw_',)
DRAW_KEYS_IN = ('strokes', 'strokePoints', 'drawn_pts')
JOINT_KEYS_IN = ('hr_cross_', 'gaze_pair_', 'joint_', 'mi_', 'kg_', 'crossmodal_',
                 'coh_', 'dtw_', 'granger_', 'convergence_', 'speech_x_gaze',
                 'gaze_xy_xcorr')


def modality_of(col):
    c = col.lower()
    if c.startswith(HR_PREFIXES) or any(k in c for k in HR_KEYS_IN):
        # If it's a joint feature (hr_cross_*) bucket as 'joint'
        if any(k in c for k in JOINT_KEYS_IN):
            return 'joint'
        return 'hr'
    if c.startswith(SPEECH_PREFIXES) or any(k in c for k in SPEECH_KEYS_IN):
        return 'speech'
    if c.startswith(DRAW_PREFIXES) or any(k in c for k in DRAW_KEYS_IN):
        return 'drawing'
    if any(k in c for k in GAZE_KEYS_IN) or c.startswith(GAZE_PREFIXES):
        if any(k in c for k in JOINT_KEYS_IN):
            return 'joint'
        return 'gaze'
    if any(k in c for k in JOINT_KEYS_IN):
        return 'joint'
    return 'other'


def select_top_in_fold(Xtr, ytr, top_k=40, corr_thresh=0.85, seed=42):
    """mRMR-style: F-test top + correlation pruning. Computed ONLY on training fold."""
    if Xtr.shape[1] <= top_k:
        return list(range(Xtr.shape[1]))
    med = np.nanmedian(Xtr, axis=0)
    Ximp = np.where(np.isnan(Xtr), med, Xtr)
    F, _ = f_classif(Ximp, ytr)
    F = np.nan_to_num(F)
    order = np.argsort(-F)
    Xs = StandardScaler().fit_transform(Ximp)
    kept = []
    for i in order:
        if len(kept) >= top_k:
            break
        if not kept:
            kept.append(i)
            continue
        corrs = [abs(np.corrcoef(Xs[:, i], Xs[:, j])[0, 1]) for j in kept]
        if max(corrs) < corr_thresh:
            kept.append(i)
    return kept


def make_stacking(seed=42):
    base = [
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                  learning_rate=0.05, random_state=seed)),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     verbose=-1, random_state=seed)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    eval_metric='logloss', verbosity=0, random_state=seed)),
        ('lr', LogisticRegression(C=1.0, penalty='l1', solver='saga',
                                     max_iter=2000, random_state=seed)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                         random_state=seed, n_jobs=1)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        n_jobs=1, passthrough=False)


def gkf_x_seeds(X, y, g, seeds=(42, 43, 44, 45, 46), n_splits=5,
                top_k=40, model='stack', return_feats=False, verbose=False):
    """5-fold GroupKFold × len(seeds) repeats with in-fold feature selection.
    Returns per-fold AUC list and pooled AUC per seed."""
    fold_aucs = []
    seed_pooled = []
    feature_log = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        # shuffle dyad order for this seed
        unique_g = np.array(sorted(set(g)))
        perm = rng.permutation(len(unique_g))
        g_perm_lookup = {dyad: i for i, dyad in enumerate(unique_g[perm])}
        g_idx = np.array([g_perm_lookup[d] for d in g])

        gkf = GroupKFold(n_splits=n_splits)
        yp = np.full(len(y), np.nan)
        for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups=g_idx)):
            if len(set(y[tr])) < 2:
                continue
            # impute training-median
            med = np.nanmedian(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            Xte = np.where(np.isnan(X[te]), med, X[te])
            # in-fold feature selection
            kept = select_top_in_fold(Xtr, y[tr], top_k=top_k, seed=sd)
            Xtr_s = Xtr[:, kept]
            Xte_s = Xte[:, kept]
            if return_feats:
                feature_log.append({'seed': sd, 'fold': fold_i, 'kept_idx': kept})
            # standardize
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr_s)
            Xte_s = sc.transform(Xte_s)
            if model == 'stack':
                m = make_stacking(sd)
            else:
                m = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                     learning_rate=0.05, random_state=sd)
            try:
                m.fit(Xtr_s, y[tr])
                yp[te] = m.predict_proba(Xte_s)[:, 1]
            except Exception as e:
                if verbose:
                    print(f"  seed {sd} fold {fold_i} failed: {e}", flush=True)
                m = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                     learning_rate=0.05, random_state=sd)
                m.fit(Xtr_s, y[tr])
                yp[te] = m.predict_proba(Xte_s)[:, 1]
            try:
                a_fold = roc_auc_score(y[te], yp[te])
                fold_aucs.append({'seed': sd, 'fold': fold_i, 'auc': a_fold,
                                    'n_test': int(len(te)), 'n_pos_test': int(y[te].sum())})
            except ValueError:
                pass
        valid = ~np.isnan(yp)
        if valid.sum() > 0:
            try:
                a_pooled = roc_auc_score(y[valid], yp[valid])
                seed_pooled.append({'seed': sd, 'pooled_auc': a_pooled})
            except ValueError:
                pass
        if verbose:
            print(f"  seed {sd}: pooled AUC = {seed_pooled[-1]['pooled_auc']:.4f}",
                  flush=True)
    return fold_aucs, seed_pooled, feature_log


def main():
    print("Loading master data...", flush=True)
    df = pd.read_csv(TRIAL)
    df['target'] = df['target_reached'].apply(to_bool)
    df['failure'] = (df['target'] == 0).astype('Int64')
    df = df[df['failure'].isin([0, 1])].copy().reset_index(drop=True)

    # Build feature pool: numeric, non-leak, non-id
    id_cols = {'dyad_id', 'session_dir', 'sessionId', 'trial', 'mapNumber',
               'target', 'failure', 'target_reached', 'clock_offset_ms',
               'trial_start_ms', 'trial_end_ms', 'trial_position', 'audio_count'}
    feat_cols = []
    leaks = []
    for c in df.columns:
        if c in id_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if is_outcome_leak(c):
            leaks.append(c)
            continue
        feat_cols.append(c)
    print(f"  Pool: {len(feat_cols)} features after excluding {len(leaks)} leaks", flush=True)

    # Drop features that are all-NaN or all-constant
    df_x = df[feat_cols].replace([np.inf, -np.inf], np.nan).copy()
    df_x = df_x.fillna(df_x.median(numeric_only=True))
    keep = [c for c in feat_cols if df_x[c].notna().all() and df_x[c].nunique() > 1]
    df_x = df_x[keep]
    feat_cols = keep
    X_full = df_x.astype(float).values
    y = df['failure'].astype(int).values
    g = df['dyad_id'].values
    print(f"  After NaN/constant filter: {X_full.shape[1]} features, "
          f"{X_full.shape[0]} trials, {int(y.sum())} positive, "
          f"{len(set(g))} dyads", flush=True)

    # Modality bucketing
    bucket = {c: modality_of(c) for c in feat_cols}
    buckets = {}
    for c, b in bucket.items():
        buckets.setdefault(b, []).append(c)
    print(f"\n  Modality buckets:", flush=True)
    for b, cols in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"    {b:<10s} {len(cols):>4d} features", flush=True)

    # ==================== A. Headline ====================
    print("\n" + "=" * 70, flush=True)
    print("A. HEADLINE: full multimodal stacking, 5-fold GKF × 5 seeds", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    fold_aucs, seed_pooled, _ = gkf_x_seeds(
        X_full, y, g, seeds=(42, 43, 44, 45, 46), n_splits=5,
        top_k=40, model='stack', verbose=True)
    print(f"  elapsed: {time.time() - t0:.0f}s", flush=True)
    headline_df = pd.DataFrame([{
        'pipeline': 'multimodal_stack',
        'n_features_pool': X_full.shape[1],
        'top_k_per_fold': 40,
        'n_folds': 5,
        'n_seeds': 5,
        'pooled_auc_mean': float(np.mean([r['pooled_auc'] for r in seed_pooled])),
        'pooled_auc_std': float(np.std([r['pooled_auc'] for r in seed_pooled], ddof=1)),
        'fold_auc_mean': float(np.mean([r['auc'] for r in fold_aucs])),
        'fold_auc_std': float(np.std([r['auc'] for r in fold_aucs], ddof=1)),
        'n_total': int(len(y)),
        'n_pos': int(y.sum()),
        'seeds': '42,43,44,45,46',
    }])
    headline_df.to_csv(OUT_DIR / 'headline_5fold_gkf.csv', index=False)
    pd.DataFrame(fold_aucs).to_csv(OUT_DIR / 'headline_fold_aucs.csv', index=False)
    pd.DataFrame(seed_pooled).to_csv(OUT_DIR / 'headline_seed_pooled.csv', index=False)
    print(f"  Headline pooled AUC: "
          f"{headline_df['pooled_auc_mean'].iloc[0]:.4f} "
          f"± {headline_df['pooled_auc_std'].iloc[0]:.4f}", flush=True)

    # ==================== B. Single-modality ====================
    print("\n" + "=" * 70, flush=True)
    print("B. SINGLE-MODALITY: stacking, 5-fold GKF × 5 seeds", flush=True)
    print("=" * 70, flush=True)
    sm_rows = []
    sm_fold_rows = []
    for mod_name, mod_cols in [
        ('hr', buckets.get('hr', [])),
        ('gaze', buckets.get('gaze', [])),
        ('speech', buckets.get('speech', [])),
        ('drawing', buckets.get('drawing', [])),
    ]:
        if len(mod_cols) < 5:
            print(f"  {mod_name}: too few features ({len(mod_cols)}); skipping", flush=True)
            continue
        idx = [feat_cols.index(c) for c in mod_cols]
        Xm = X_full[:, idx]
        print(f"\n  {mod_name}: {Xm.shape[1]} features", flush=True)
        t0 = time.time()
        fold_aucs_m, seed_pooled_m, _ = gkf_x_seeds(
            Xm, y, g, seeds=(42, 43, 44, 45, 46), n_splits=5,
            top_k=min(40, Xm.shape[1]), model='stack', verbose=True)
        print(f"    elapsed: {time.time() - t0:.0f}s", flush=True)
        sm_rows.append({
            'modality': mod_name,
            'n_features_pool': Xm.shape[1],
            'top_k_per_fold': min(40, Xm.shape[1]),
            'pooled_auc_mean': float(np.mean([r['pooled_auc'] for r in seed_pooled_m])),
            'pooled_auc_std': float(np.std([r['pooled_auc'] for r in seed_pooled_m], ddof=1)),
            'fold_auc_mean': float(np.mean([r['auc'] for r in fold_aucs_m])),
            'fold_auc_std': float(np.std([r['auc'] for r in fold_aucs_m], ddof=1)),
        })
        for r in fold_aucs_m:
            r['modality'] = mod_name
            sm_fold_rows.append(r)
    pd.DataFrame(sm_rows).to_csv(OUT_DIR / 'single_modality_5fold_gkf.csv', index=False)
    pd.DataFrame(sm_fold_rows).to_csv(OUT_DIR / 'single_modality_fold_aucs.csv', index=False)
    print(f"\n  Single-modality summary written.", flush=True)

    # ==================== C. Modality ablation (D / M / J / D+M / D+M+J etc.) ====================
    print("\n" + "=" * 70, flush=True)
    print("C. MODALITY ABLATION: HistGB only, 5-fold GKF × 3 seeds", flush=True)
    print("=" * 70, flush=True)
    # D / M assignment by column-name keyword
    D_KEYS = ('_d_', '_d.', '_director', 'director_', '_d_z', '_d_mean')
    M_KEYS = ('_m_', '_m.', '_matcher', 'matcher_', '_m_z', '_m_mean')
    d_cols, m_cols, j_cols = [], [], []
    for c in feat_cols:
        cl = c.lower()
        is_joint = bucket[c] == 'joint'
        if is_joint:
            j_cols.append(c)
            continue
        # Per-operator marginal
        is_d = any(k in cl for k in D_KEYS) or '_d_' in cl or 'director' in cl
        is_m = any(k in cl for k in M_KEYS) or '_m_' in cl or 'matcher' in cl
        if is_d and not is_m:
            d_cols.append(c)
        elif is_m and not is_d:
            m_cols.append(c)
        else:
            # ambiguous (no operator suffix) — split by content (drawing/speech/gaze)
            # for simplicity, assign to whichever has stronger keyword
            if 'matcher' in cl or '_m_' in cl:
                m_cols.append(c)
            elif 'director' in cl or '_d_' in cl:
                d_cols.append(c)
    print(f"  Director-only: {len(d_cols)} | Matcher-only: {len(m_cols)} | "
          f"Joint-only: {len(j_cols)}", flush=True)

    configs = [
        ('Director-only', d_cols),
        ('Matcher-only', m_cols),
        ('Joint-only', j_cols),
        ('Director + Joint', d_cols + j_cols),
        ('Matcher + Joint', m_cols + j_cols),
        ('Director + Matcher', d_cols + m_cols),
        ('Multimodal (D+M+J)', d_cols + m_cols + j_cols),
    ]
    abl_rows = []
    for name, cols in configs:
        if len(cols) < 5:
            continue
        idx = [feat_cols.index(c) for c in cols]
        Xc = X_full[:, idx]
        t0 = time.time()
        fold_aucs_c, seed_pooled_c, _ = gkf_x_seeds(
            Xc, y, g, seeds=(42, 43, 44), n_splits=5,
            top_k=min(40, Xc.shape[1]), model='hgb', verbose=False)
        abl_rows.append({
            'config': name,
            'n_features_pool': Xc.shape[1],
            'top_k_per_fold': min(40, Xc.shape[1]),
            'pooled_auc_mean': float(np.mean([r['pooled_auc'] for r in seed_pooled_c])),
            'pooled_auc_std': float(np.std([r['pooled_auc'] for r in seed_pooled_c], ddof=1))
                if len(seed_pooled_c) > 1 else 0.0,
            'fold_auc_mean': float(np.mean([r['auc'] for r in fold_aucs_c])),
            'fold_auc_std': float(np.std([r['auc'] for r in fold_aucs_c], ddof=1)),
        })
        print(f"  {name:<22s}: "
              f"AUC = {abl_rows[-1]['pooled_auc_mean']:.4f} "
              f"± {abl_rows[-1]['pooled_auc_std']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(abl_rows).to_csv(OUT_DIR / 'ablation_5fold_gkf.csv', index=False)

    # ==================== D. Permutation null ====================
    print("\n" + "=" * 70, flush=True)
    print("D. PERMUTATION NULL: full pool, HistGB, 5-fold GKF, 200 perms", flush=True)
    print("=" * 70, flush=True)

    # Real-labels AUC (single seed for speed, matching fix10's protocol)
    t0 = time.time()
    fold_aucs_real, seed_pooled_real, _ = gkf_x_seeds(
        X_full, y, g, seeds=(42,), n_splits=5,
        top_k=40, model='hgb', verbose=False)
    real_auc = seed_pooled_real[0]['pooled_auc'] if seed_pooled_real else float('nan')
    print(f"  Real-labels AUC: {real_auc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # Permutation null: shuffle within-dyad labels
    print(f"  Running {200} permutations...", flush=True)
    rng = np.random.default_rng(42)
    dyads = sorted(set(g))
    null_aucs = []
    t0 = time.time()
    for it in range(200):
        y_perm = y.copy()
        for d in dyads:
            mask = (g == d)
            ys = y[mask]
            ys_p = rng.permutation(ys)
            y_perm[mask] = ys_p
        try:
            _, seed_pooled_p, _ = gkf_x_seeds(
                X_full, y_perm, g, seeds=(42,), n_splits=5,
                top_k=40, model='hgb', verbose=False)
            if seed_pooled_p:
                null_aucs.append(seed_pooled_p[0]['pooled_auc'])
        except Exception:
            continue
        if (it + 1) % 20 == 0:
            print(f"    perm {it+1}/200: null median so far = "
                  f"{np.median(null_aucs):.4f}  "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    null_arr = np.array(null_aucs)
    pval = float((null_arr >= real_auc).mean()) if len(null_arr) > 0 else float('nan')
    perm_df = pd.DataFrame([{
        'auc_empirical': real_auc,
        'null_median': float(np.median(null_arr)) if len(null_arr) > 0 else float('nan'),
        'null_p95': float(np.percentile(null_arr, 95)) if len(null_arr) > 0 else float('nan'),
        'p_value': pval,
        'n_perm': int(len(null_arr)),
        'cv': '5-fold GroupKFold',
        'model': 'HistGB single',
        'top_k': 40,
        'feature_pool_size': int(X_full.shape[1]),
    }])
    perm_df.to_csv(OUT_DIR / 'permutation_null_5fold_gkf.csv', index=False)
    pd.DataFrame({'null_auc': null_arr}).to_csv(OUT_DIR / 'permutation_null_distribution.csv',
                                                   index=False)
    print(f"\n  Permutation null:\n"
          f"    real AUC      = {real_auc:.4f}\n"
          f"    null median   = {perm_df['null_median'].iloc[0]:.4f}\n"
          f"    null p95      = {perm_df['null_p95'].iloc[0]:.4f}\n"
          f"    empirical p   = {pval:.4f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PHASE 50 COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Outputs written to: {OUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
