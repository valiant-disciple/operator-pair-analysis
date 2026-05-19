#!/usr/bin/env python3
"""
Phase 2A: Map-adjusted outcomes + composite outcomes + within-dyad rank.

Adds these to master_clean.csv:
  - map_difficulty_chamfer: mean Chamfer for this map across all dyads (empirical difficulty)
  - map_difficulty_iou: same for IoU
  - map_difficulty_target_rate: target_reached success rate per map
  - chamfer_map_adjusted: this trial's Chamfer minus map mean (isolates dyad effort)
  - chamfer_within_dyad_rank: rank of this trial's Chamfer among dyad's 6 trials (1=worst, 6=best)
  - chamfer_within_dyad_z: z-score of this trial's Chamfer relative to dyad's other trials
  - composite_accuracy_pca1: first PC of (Chamfer, IoU, F1, boundary_F1, target_reached, path_confidence)
  - composite_accuracy_pca1_explained: % variance explained by first PC
"""
import csv
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DSA_ROOT = Path("/Users/kolosus/Documents/DSA")
BATCH_OUT = DSA_ROOT / "analysis" / "batch_out"
MASTER = BATCH_OUT / "master_clean.csv"


def to_bool(s):
    if pd.isna(s): return float('nan')
    if isinstance(s, bool): return float(s)
    s = str(s).strip().lower()
    if s in ('true','1','yes'): return 1.0
    if s in ('false','0','no'): return 0.0
    return float('nan')


def main():
    df = pd.read_csv(MASTER)
    print(f"Loaded {len(df)} rows")

    # Convert outcomes to numeric
    for c in ['map_chamfer', 'map_iou', 'map_f1', 'map_boundary_f1', 'path_confidence', 'mapNumber']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['target_reached_num'] = df['target_reached'].apply(to_bool)

    # ── Map difficulty: empirical mean per map ──
    print("\n=== Map difficulty ===")
    map_stats = df.groupby('mapNumber').agg(
        map_difficulty_chamfer=('map_chamfer', 'mean'),
        map_difficulty_iou=('map_iou', 'mean'),
        map_difficulty_boundary_f1=('map_boundary_f1', 'mean'),
        map_difficulty_target_rate=('target_reached_num', 'mean'),
        map_difficulty_n_trials=('map_chamfer', 'count'),
    )
    print(map_stats.to_string())

    # Save map difficulty table
    map_stats.to_csv(BATCH_OUT / 'map_difficulty.csv')

    # Merge back to df
    df = df.merge(map_stats[['map_difficulty_chamfer', 'map_difficulty_iou',
                             'map_difficulty_boundary_f1', 'map_difficulty_target_rate']],
                  left_on='mapNumber', right_index=True, how='left')

    # ── Map-adjusted outcomes ──
    df['chamfer_map_adjusted'] = df['map_chamfer'] - df['map_difficulty_chamfer']
    df['iou_map_adjusted'] = df['map_iou'] - df['map_difficulty_iou']
    df['boundary_f1_map_adjusted'] = df['map_boundary_f1'] - df['map_difficulty_boundary_f1']

    # ── Within-dyad rank and z-score ──
    df['chamfer_within_dyad_rank'] = df.groupby('dyad_id')['map_chamfer'].rank(method='dense')
    df['chamfer_within_dyad_z'] = df.groupby('dyad_id')['map_chamfer'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)

    # ── Composite accuracy via PCA ──
    print("\n=== Composite outcome (PCA) ===")
    components = ['map_chamfer', 'map_iou', 'map_f1', 'map_boundary_f1',
                  'target_reached_num', 'path_confidence']
    valid_cols = [c for c in components if c in df.columns]
    sub = df[valid_cols].copy()

    # Direction: lower Chamfer = better, higher IoU/F1 = better, higher target_reached = better
    # Negate Chamfer so higher = better for all
    if 'map_chamfer' in sub.columns:
        sub['map_chamfer'] = -sub['map_chamfer']

    # Drop rows with too many missing
    valid = sub.notna().sum(axis=1) >= 4
    sub_v = sub[valid]
    print(f"  Rows with ≥4 valid outcome components: {len(sub_v)}")

    imp = SimpleImputer(strategy='median')
    sub_imp = imp.fit_transform(sub_v.values)
    scaler = StandardScaler()
    sub_z = scaler.fit_transform(sub_imp)

    pca = PCA(n_components=2)
    pca_scores = pca.fit_transform(sub_z)
    print(f"  PCA1 explained variance: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PCA2 explained variance: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  PCA1 loadings:")
    for c, w in zip(valid_cols, pca.components_[0]):
        print(f"    {c}: {w:+.3f}")

    # Add PCA scores back (negated if loadings indicate higher PC = lower accuracy)
    df['composite_accuracy_pca1'] = float('nan')
    df.loc[valid, 'composite_accuracy_pca1'] = pca_scores[:, 0]
    df['composite_accuracy_pca2'] = float('nan')
    df.loc[valid, 'composite_accuracy_pca2'] = pca_scores[:, 1]

    df.to_csv(MASTER, index=False)
    print(f"\nWrote {MASTER}")
    print(f"New columns: map_difficulty_*, chamfer_map_adjusted, chamfer_within_dyad_rank, "
          f"chamfer_within_dyad_z, composite_accuracy_pca1, composite_accuracy_pca2")


if __name__ == '__main__':
    main()
