#!/usr/bin/env python3
"""
Phase 7B: Benchmark multiple models on the passive multimodal task.

Models tested:
  - HistGradientBoosting (current default)
  - XGBoost
  - LightGBM
  - RandomForest
  - ExtraTrees
  - Logistic Regression (L2)
  - SVM (RBF kernel)
  - Stacking ensemble (HGB + XGB + LGB → logistic meta)

Outcome: target_reached (binary).
Feature set: best passive (1A + 1B + 2B + 3A + 7A).

LOO-dyad CV.
"""
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

BATCH_OUT = Path("/Users/kolosus/Documents/DSA/analysis/batch_out")
MASTER = BATCH_OUT / "master_clean.csv"


def to_bool(s):
    if pd.isna(s): return float('nan')
    if isinstance(s, bool): return float(s)
    s = str(s).strip().lower()
    if s in ('true','1','yes'): return 1.0
    if s in ('false','0','no'): return 0.0
    return float('nan')


EXCLUDE = {'dyad_id','session_dir','trial','trial_position','sessionId','sessionId_director','sessionId_matcher',
    'mapNumber','trial_start_ms','trial_start_iso','trial_end_ms','trial_end_iso','director_note','ref_clock',
    'audio_count','clock_offset_ms','target_reached_num','demo_gender_pair','demo_gender_d','demo_gender_m',
    'target', 'target_reached'}
OUTCOMES = {'map_iou','map_f1','map_chamfer','map_hausdorff','map_dice','map_ssim','map_boundary_f1',
    'map_boundary_p','map_boundary_r','map_precision','map_recall','coverage_gt','coverage_pred',
    'target_reached','path_confidence','chamfer_map_adjusted','iou_map_adjusted','boundary_f1_map_adjusted',
    'chamfer_within_dyad_rank','chamfer_within_dyad_z','composite_accuracy_pca1','composite_accuracy_pca2',
    'map_difficulty_chamfer','map_difficulty_iou','map_difficulty_boundary_f1','map_difficulty_target_rate',
    'n_landmarks','landmarks_covered','landmark_coverage_rate','landmark_order_correct',
    'landmark_accuracy','route_coverage','drawn_on_route_pct'}


def is_passive_full(c):
    """Best passive feature set: 1A + 1B + 2B + 3A + 7A (no drawing, no subjective)."""
    if c.startswith('draw_') or c in ('strokes', 'strokePoints'): return False
    if c.startswith('tlx_') or c.startswith('psmm_'): return False  # exclude subjective except psmm_dir_mat
    if c.startswith('demo_'): return False
    return True  # rest is passive multimodal


def make_models():
    return {
        'HistGB': HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='auc', verbosity=0),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=2),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=2),
        'LogReg-L2': Pipeline([('scaler', StandardScaler()),
                               ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=42))]),
        'SVM-RBF': Pipeline([('scaler', StandardScaler()),
                             ('clf', SVC(kernel='rbf', probability=True, random_state=42, C=1.0))]),
    }


def evaluate(model_factory, X, y, groups):
    valid = ~np.isnan(y)
    if valid.sum() < 30: return float('nan'), float('nan')
    Xv = X[valid]; yv = y[valid]; gv = groups[valid]
    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(yv, dtype=float)
    for tr_idx, te_idx in logo.split(Xv, yv, gv):
        if len(set(yv[tr_idx])) < 2: continue
        m = model_factory()
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(Xv[tr_idx])
        Xte = imp.transform(Xv[te_idx])
        m.fit(Xtr, yv[tr_idx])
        try:
            y_pred[te_idx] = m.predict_proba(Xte)[:, 1]
        except Exception:
            y_pred[te_idx] = m.decision_function(Xte) if hasattr(m, 'decision_function') else 0.5
    return roc_auc_score(yv, y_pred), average_precision_score(yv, y_pred)


def stacking_ensemble():
    estimators = [
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='auc', verbosity=0)),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5, n_jobs=2
    )


def main():
    df = pd.read_csv(MASTER)
    for c in df.columns:
        if c not in EXCLUDE:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['target'] = df['target_reached'].apply(to_bool)
    y = df['target'].values
    groups = df['dyad_id'].values

    cols_all = [c for c in df.columns if c not in EXCLUDE and c not in OUTCOMES]
    passive_cols = [c for c in cols_all if is_passive_full(c)]
    passive_cols = [c for c in passive_cols if not df[c].isna().all()]
    print(f"Passive features: {len(passive_cols)}")

    X = df[passive_cols].values.astype(float)
    # Clip infinities (DTW edge cases) to NaN
    X = np.where(np.isinf(X), np.nan, X)

    print(f"\n{'Model':<20} {'AUC':>7} {'AP':>7}")
    print('-' * 40)

    factories = make_models()
    results = []
    for name, model in factories.items():
        try:
            auc, ap = evaluate(lambda m=model: type(m)(**(m.get_params() if hasattr(m, 'get_params') else {})) if not isinstance(m, Pipeline) else Pipeline(m.steps), X, y, groups)
            # Re-create properly using the model factory
            def factory(model=model):
                # Clone the model
                from sklearn.base import clone
                return clone(model)
            auc, ap = evaluate(factory, X, y, groups)
            print(f"{name:<20} {auc:>7.3f} {ap:>7.3f}")
            results.append({'model': name, 'auc': float(auc), 'ap': float(ap)})
        except Exception as e:
            print(f"{name:<20} FAILED: {e}")
            results.append({'model': name, 'auc': float('nan'), 'ap': float('nan'), 'error': str(e)})

    # Stacking
    print("\nTraining stacking ensemble...")
    try:
        from sklearn.base import clone
        auc, ap = evaluate(lambda: clone(stacking_ensemble()), X, y, groups)
        print(f"{'Stacking':<20} {auc:>7.3f} {ap:>7.3f}")
        results.append({'model': 'Stacking', 'auc': float(auc), 'ap': float(ap)})
    except Exception as e:
        print(f"Stacking FAILED: {e}")
        results.append({'model': 'Stacking', 'auc': float('nan'), 'ap': float('nan'), 'error': str(e)})

    pd.DataFrame(results).to_csv(BATCH_OUT / 'multi_model_benchmark.csv', index=False)
    print(f"\nWrote {BATCH_OUT / 'multi_model_benchmark.csv'}")


if __name__ == '__main__':
    main()
