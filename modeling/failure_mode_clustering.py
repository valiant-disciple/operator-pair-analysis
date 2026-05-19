#!/usr/bin/env python3
"""
Phase 6C: Failure mode taxonomy via clustering.

Approach: For each failure trial (bottom-tertile Chamfer per-dyad), extract a
multimodal "fingerprint" vector. Cluster fingerprints to identify distinct
failure modes.

Fingerprint dimensions:
  - Director workload (W_workload_d)
  - Matcher workload (W_workload_m)
  - Workload asymmetry (gini)
  - HR cross-CRQA DET (synchrony)
  - Joint attention proportion
  - Director gaze entropy
  - Matcher gaze entropy
  - Mid-trial pupil dilation (d)
  - Late-window HR change (d, m)

We expect clusters like:
  - "Workload overload" (high W_d + high W_m, sync drops)
  - "Communication breakdown" (low joint attention, high gaze entropy)
  - "Disengagement" (low pupil, low HR variability)
  - "Asymmetric load" (high W_d, low W_m or vice versa)
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BATCH_OUT = Path("/Users/kolosus/Documents/DSA/analysis/batch_out")
MASTER = BATCH_OUT / "master_clean.csv"


def main():
    df = pd.read_csv(MASTER)
    for c in df.columns:
        if c not in ('dyad_id', 'session_dir', 'target_reached'):
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Identify failure trials: bottom tertile per dyad
    df['rank'] = df.groupby('dyad_id')['map_chamfer'].rank(method='dense')
    df['n_per_dyad'] = df.groupby('dyad_id')['map_chamfer'].transform('count')
    df['is_failure'] = df.apply(
        lambda r: 1.0 if r['rank'] >= r['n_per_dyad'] - 1 else 0.0, axis=1)

    fail_df = df[df['is_failure'] == 1].copy()
    print(f"Failure trials: {len(fail_df)} out of {len(df)} total")

    # Define fingerprint features (all multimodal, no surveys, no drawing)
    fingerprint_cols = [
        'W_workload_d', 'W_workload_m',
        'hr_cross_crqa_det', 'hr_cross_crqa_lam',
        'gaze_pair_gaze_conv_pct_within_100px',
        'gaze_director_scanpath_entropy', 'gaze_matcher_scanpath_entropy',
        'gaze_director_pupil_mean', 'gaze_matcher_pupil_mean',
        'pre_mid_pupil_d_mean', 'pre_mid_hr_d_mean',
        'pre_late_minus_early_hr_sync',
        'hr_director_rmssd_ms', 'hr_matcher_rmssd_ms',
        'hr_director_bpm_mean', 'hr_matcher_bpm_mean',
        'speech_x_gaze_d',
    ]

    # Use available
    avail = [c for c in fingerprint_cols if c in fail_df.columns and fail_df[c].notna().sum() >= 30]
    print(f"\nFingerprint dimensions ({len(avail)}):")
    for c in avail:
        print(f"  {c}: n_valid={fail_df[c].notna().sum()}")

    if len(avail) < 5:
        print("Insufficient features for clustering")
        return

    X = fail_df[avail].values

    # Impute and standardize
    imp = SimpleImputer(strategy='median')
    Xi = imp.fit_transform(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xi)

    # Choose k via silhouette
    print('\n## Choosing k:')
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"  k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.1f}")

    # Try k=3 and k=4
    for k_try in [3, 4]:
        print(f'\n## k={k_try} clustering:')
        km = KMeans(n_clusters=k_try, random_state=42, n_init=20)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"  Silhouette: {sil:.4f}")

        fail_df[f'cluster_k{k_try}'] = labels

        # Cluster centroids in z-score space
        print(f"  Cluster centroids (z-scored fingerprint dimensions):")
        centroids = pd.DataFrame(km.cluster_centers_, columns=avail)
        # Show top 5 distinguishing features per cluster
        for c in range(k_try):
            print(f"\n  Cluster {c} (n={int((labels == c).sum())}):")
            cv = centroids.iloc[c].sort_values(key=lambda x: x.abs(), ascending=False)
            for feat in cv.head(8).index:
                z = cv[feat]
                direction = '↑' if z > 0 else '↓'
                print(f"    {direction} {feat:<45} z={z:+.2f}")

        # Outcome differences per cluster
        print(f"\n  Outcome by cluster:")
        for c in range(k_try):
            members = fail_df[fail_df[f'cluster_k{k_try}'] == c]
            print(f"    Cluster {c}: chamfer mean={members['map_chamfer'].mean():.1f}, "
                  f"target_reached %={members['target_reached'].apply(lambda x: 1.0 if str(x).lower() in ('true','1') else 0.0).mean()*100:.0f}, "
                  f"path_conf={members['path_confidence'].mean():.2f}")

    # Save
    fail_df_save = fail_df[['dyad_id', 'trial', 'mapNumber', 'map_chamfer', 'target_reached',
                             'cluster_k3', 'cluster_k4'] + avail]
    fail_df_save.to_csv(BATCH_OUT / 'failure_modes.csv', index=False)
    print(f"\nWrote {BATCH_OUT / 'failure_modes.csv'}")


if __name__ == '__main__':
    main()
