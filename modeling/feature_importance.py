#!/usr/bin/env python3
"""
SHAP feature importance for the FULL multimodal model (M10).
Trained on all data (no CV) for stable SHAP values, then computes:
  - Per-feature mean |SHAP| value
  - Top 30 features
  - Per-modality summed importance

Outputs:
  - shap_top_features.csv (feature, mean_abs_shap, modality)
  - modality_importance.csv (modality, total importance, count of features)
"""
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import math
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

BATCH_OUT = Path("/Users/kolosus/Documents/DSA/analysis/batch_out")
MASTER = BATCH_OUT / "master_clean.csv"

EXCLUDE_COLS = {
    'dyad_id', 'session_dir', 'trial', 'trial_position', 'sessionId',
    'sessionId_director', 'sessionId_matcher', 'mapNumber',
    'trial_start_ms', 'trial_start_iso', 'trial_end_ms', 'trial_end_iso',
    'director_note', 'ref_clock', 'audio_count', 'clock_offset_ms',
    'map_iou', 'map_f1', 'map_chamfer', 'map_hausdorff', 'map_dice', 'map_ssim',
    'map_boundary_f1', 'map_boundary_p', 'map_boundary_r', 'map_precision', 'map_recall',
    'coverage_gt', 'coverage_pred', 'target_reached', 'path_confidence',
}


def assign_modality(col):
    if col.startswith('hr_'): return 'HR/HRV'
    if col.startswith('gaze_'): return 'Gaze'
    if col.startswith('prosody_'): return 'Speech'
    if col.startswith('draw_') or col.startswith('strokes') or col == 'strokePoints': return 'Drawing'
    if col.startswith('tlx_'): return 'TLX'
    if col.startswith('psmm_'): return 'PSMM'
    return 'Other'


def to_float(s):
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return float('nan')
        return v
    except (ValueError, TypeError):
        return float('nan')


def main():
    df = pd.read_csv(MASTER)
    print(f"Loaded {len(df)} rows × {df.shape[1]} cols")

    # Convert all to numeric except metadata
    for c in df.columns:
        if c in ('dyad_id', 'session_dir', 'sessionId'):
            continue
        df[c] = df[c].apply(to_float)

    df = df.dropna(subset=['map_chamfer'])
    print(f"After dropping NaN outcome: {len(df)}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # Drop columns with all-NaN BEFORE imputing
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    X = df[feature_cols].values.astype(float)
    y = df['map_chamfer'].values

    # Impute
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    print(f"Training HistGradientBoosting on {X_imp.shape[0]} rows × {X_imp.shape[1]} features")
    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=4, learning_rate=0.05,
        random_state=42
    )
    model.fit(X_imp, y)

    print("Computing SHAP values (this may take a few minutes)...")
    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Per-feature
    feat_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
        'modality': [assign_modality(c) for c in feature_cols],
    }).sort_values('mean_abs_shap', ascending=False)

    feat_importance.to_csv(BATCH_OUT / 'shap_top_features.csv', index=False)
    print("\nTop 30 features by SHAP importance:")
    print(feat_importance.head(30).to_string(index=False))

    # Per-modality summed
    modality_importance = feat_importance.groupby('modality').agg(
        total_importance=('mean_abs_shap', 'sum'),
        n_features=('feature', 'count'),
        max_feature=('feature', 'first'),
        max_shap=('mean_abs_shap', 'max'),
    ).sort_values('total_importance', ascending=False)

    modality_importance.to_csv(BATCH_OUT / 'modality_importance.csv')
    print("\nModality-level importance:")
    print(modality_importance)


if __name__ == '__main__':
    main()
