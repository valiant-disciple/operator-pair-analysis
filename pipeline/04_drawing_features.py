#!/usr/bin/env python3
"""
Drawing behavior feature extraction from stroke data.

Computes temporal, kinematic, behavioral, spatial, and complexity features
from Matcher's stroke/point-level drawing data.

Usage:
    from drawing_features import extract_drawing_features

Input: list of stroke dicts from strokes.json, each with:
    {points: [{x, y, t}, ...], mode: "draw"|"erase", width: N, ...}
"""

import math
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


def _sf(v, default=None):
    if v is None or v == "":
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _point_arrays(stroke: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (x, y, t_ms) arrays from a stroke's points."""
    xs, ys, ts = [], [], []
    for p in stroke.get("points", []):
        x = _sf(p.get("x"))
        y = _sf(p.get("y"))
        t = _sf(p.get("t"))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
            ts.append(t if t is not None else 0)
    return np.array(xs), np.array(ys), np.array(ts)


def _stroke_length(xs: np.ndarray, ys: np.ndarray) -> float:
    """Total Euclidean length of a polyline."""
    if len(xs) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))


def _stroke_displacement(xs: np.ndarray, ys: np.ndarray) -> float:
    """Straight-line distance from first to last point."""
    if len(xs) < 2:
        return 0.0
    return float(math.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))


def _curvature(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute discrete curvature at each interior point."""
    if len(xs) < 3:
        return np.array([])
    dx = np.diff(xs)
    dy = np.diff(ys)
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    num = np.abs(dx[:-1] * ddy - dy[:-1] * ddx)
    denom = (dx[:-1] ** 2 + dy[:-1] ** 2) ** 1.5
    denom = np.where(denom > 1e-8, denom, 1e-8)
    return num / denom


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def temporal_features(strokes: List[Dict], trial_duration_ms: float = 210000) -> Dict[str, float]:
    """Temporal drawing behavior features."""
    if not strokes:
        return {}

    draw_strokes = [s for s in strokes if s.get("mode", "draw") == "draw"]
    all_strokes = strokes

    # Collect all point timestamps
    all_ts = []
    stroke_start_ts = []
    stroke_end_ts = []
    for s in all_strokes:
        pts = s.get("points", [])
        pts_t = [_sf(p.get("t")) for p in pts]
        pts_t = [t for t in pts_t if t is not None and t > 0]
        if pts_t:
            stroke_start_ts.append(min(pts_t))
            stroke_end_ts.append(max(pts_t))
            all_ts.extend(pts_t)

    if not all_ts:
        return {}

    t_min = min(all_ts)
    t_max = max(all_ts)
    total_drawing_span = t_max - t_min

    feats = {}

    # Time to first stroke (from trial start = t=0 in relative terms)
    # Use the earliest point timestamp
    feats["dt_first_stroke_ms"] = stroke_start_ts[0] - t_min if len(stroke_start_ts) > 1 else 0.0

    # Total active drawing time (sum of stroke durations)
    stroke_durations = []
    for s_start, s_end in zip(stroke_start_ts, stroke_end_ts):
        dur = s_end - s_start
        if dur > 0:
            stroke_durations.append(dur)

    total_active_ms = sum(stroke_durations) if stroke_durations else 0.0
    feats["dt_active_drawing_ms"] = total_active_ms
    feats["dt_drawing_duty_cycle"] = total_active_ms / trial_duration_ms if trial_duration_ms > 0 else 0.0

    # Stroke durations
    if stroke_durations:
        durs = np.array(stroke_durations)
        feats["dt_stroke_dur_mean"] = float(np.mean(durs))
        feats["dt_stroke_dur_median"] = float(np.median(durs))
        feats["dt_stroke_dur_std"] = float(np.std(durs, ddof=1)) if len(durs) > 1 else 0.0
        feats["dt_stroke_dur_max"] = float(np.max(durs))
        feats["dt_stroke_dur_min"] = float(np.min(durs))

    # Inter-stroke intervals (gaps between consecutive strokes)
    if len(stroke_start_ts) > 1 and len(stroke_end_ts) > 1:
        intervals = []
        sorted_pairs = sorted(zip(stroke_start_ts, stroke_end_ts))
        for i in range(1, len(sorted_pairs)):
            gap = sorted_pairs[i][0] - sorted_pairs[i - 1][1]
            if gap > 0:
                intervals.append(gap)

        if intervals:
            gaps = np.array(intervals)
            feats["dt_interstroke_mean"] = float(np.mean(gaps))
            feats["dt_interstroke_median"] = float(np.median(gaps))
            feats["dt_interstroke_std"] = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
            feats["dt_interstroke_max"] = float(np.max(gaps))

            # Hesitations: inter-stroke gaps > 3 seconds
            hesitations = [g for g in intervals if g > 3000]
            feats["dt_hesitation_count"] = len(hesitations)
            feats["dt_hesitation_total_ms"] = sum(hesitations)

    # Drawing pace over time (points per 30s window)
    if total_drawing_span > 0:
        window_ms = 30000
        n_windows = max(1, int(total_drawing_span / window_ms))
        points_per_window = []
        for w in range(n_windows):
            w_start = t_min + w * window_ms
            w_end = w_start + window_ms
            count = sum(1 for t in all_ts if w_start <= t < w_end)
            points_per_window.append(count)
        if len(points_per_window) > 1:
            feats["dt_pace_trend"] = float(np.polyfit(
                range(len(points_per_window)), points_per_window, 1)[0])
        feats["dt_pace_mean"] = float(np.mean(points_per_window))

    return feats


def kinematic_features(strokes: List[Dict]) -> Dict[str, float]:
    """Kinematic drawing features: speed, acceleration, curvature."""
    if not strokes:
        return {}

    draw_strokes = [s for s in strokes if s.get("mode", "draw") == "draw"]
    if not draw_strokes:
        return {}

    all_speeds = []
    all_accels = []
    all_curvatures = []
    all_lengths = []
    all_displacements = []
    all_straightness = []
    all_point_densities = []

    for s in draw_strokes:
        xs, ys, ts = _point_arrays(s)
        if len(xs) < 2:
            continue

        length = _stroke_length(xs, ys)
        displacement = _stroke_displacement(xs, ys)
        all_lengths.append(length)
        all_displacements.append(displacement)

        # Straightness ratio (1.0 = perfectly straight)
        straightness = displacement / length if length > 0 else 1.0
        all_straightness.append(straightness)

        # Point density (points per pixel of length)
        all_point_densities.append(len(xs) / length if length > 0 else 0.0)

        # Speed between consecutive points
        if ts.any() and len(ts) > 1:
            dt = np.diff(ts) / 1000.0  # seconds
            dx = np.diff(xs)
            dy = np.diff(ys)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            valid = dt > 0.001  # avoid division by zero
            if valid.any():
                speeds = dist[valid] / dt[valid]
                all_speeds.extend(speeds.tolist())

                # Acceleration
                if len(speeds) > 1:
                    dt_speed = dt[valid][:-1]
                    dv = np.diff(speeds)
                    accel_valid = dt_speed > 0.001
                    if accel_valid.any():
                        accels = dv[accel_valid] / dt_speed[accel_valid]
                        all_accels.extend(accels.tolist())

        # Curvature
        curv = _curvature(xs, ys)
        if len(curv) > 0:
            all_curvatures.extend(curv.tolist())

    feats = {}

    # Stroke lengths
    if all_lengths:
        lens = np.array(all_lengths)
        feats["dk_stroke_len_mean"] = float(np.mean(lens))
        feats["dk_stroke_len_median"] = float(np.median(lens))
        feats["dk_stroke_len_std"] = float(np.std(lens, ddof=1)) if len(lens) > 1 else 0.0
        feats["dk_stroke_len_total"] = float(np.sum(lens))
        feats["dk_stroke_len_max"] = float(np.max(lens))
        feats["dk_stroke_len_min"] = float(np.min(lens))

    # Displacements
    if all_displacements:
        feats["dk_displacement_mean"] = float(np.mean(all_displacements))
        feats["dk_displacement_total"] = float(np.sum(all_displacements))

    # Straightness
    if all_straightness:
        feats["dk_straightness_mean"] = float(np.mean(all_straightness))
        feats["dk_straightness_std"] = float(np.std(all_straightness)) if len(all_straightness) > 1 else 0.0

    # Point density
    if all_point_densities:
        feats["dk_point_density_mean"] = float(np.mean(all_point_densities))

    # Speed
    if all_speeds:
        spd = np.array(all_speeds)
        feats["dk_speed_mean"] = float(np.mean(spd))
        feats["dk_speed_median"] = float(np.median(spd))
        feats["dk_speed_std"] = float(np.std(spd, ddof=1)) if len(spd) > 1 else 0.0
        feats["dk_speed_max"] = float(np.max(spd))
        feats["dk_speed_p25"] = float(np.percentile(spd, 25))
        feats["dk_speed_p75"] = float(np.percentile(spd, 75))

    # Acceleration
    if all_accels:
        acc = np.array(all_accels)
        feats["dk_accel_mean"] = float(np.mean(acc))
        feats["dk_accel_std"] = float(np.std(acc))
        feats["dk_accel_max"] = float(np.max(np.abs(acc)))

    # Curvature
    if all_curvatures:
        curv = np.array(all_curvatures)
        # Clip extreme curvatures (numerical artifacts)
        curv = curv[curv < np.percentile(curv, 99)]
        if len(curv) > 0:
            feats["dk_curvature_mean"] = float(np.mean(curv))
            feats["dk_curvature_median"] = float(np.median(curv))
            feats["dk_curvature_std"] = float(np.std(curv))
            feats["dk_curvature_max"] = float(np.max(curv))

    return feats


def behavioral_features(strokes: List[Dict]) -> Dict[str, float]:
    """Behavioral drawing features: stroke count, erase ratio, backtracking."""
    if not strokes:
        return {}

    draw_strokes = [s for s in strokes if s.get("mode", "draw") == "draw"]
    erase_strokes = [s for s in strokes if s.get("mode") == "erase"]

    n_draw = len(draw_strokes)
    n_erase = len(erase_strokes)
    n_total = len(strokes)

    feats = {
        "db_stroke_count": n_total,
        "db_draw_count": n_draw,
        "db_erase_count": n_erase,
        "db_erase_ratio": n_erase / n_total if n_total > 0 else 0.0,
    }

    # Total points
    total_draw_pts = sum(len(s.get("points", [])) for s in draw_strokes)
    total_erase_pts = sum(len(s.get("points", [])) for s in erase_strokes)
    feats["db_draw_points"] = total_draw_pts
    feats["db_erase_points"] = total_erase_pts
    feats["db_total_points"] = total_draw_pts + total_erase_pts

    # Stroke fragmentation: more strokes for same total length = more fragmented
    total_len = 0
    for s in draw_strokes:
        xs, ys, _ = _point_arrays(s)
        total_len += _stroke_length(xs, ys)
    feats["db_fragmentation"] = n_draw / total_len if total_len > 0 else 0.0

    # Erase-then-redraw patterns: count erase strokes followed by draw strokes
    erase_redraw = 0
    for i in range(len(strokes) - 1):
        if strokes[i].get("mode") == "erase" and strokes[i + 1].get("mode", "draw") == "draw":
            erase_redraw += 1
    feats["db_erase_redraw_count"] = erase_redraw

    # Width usage diversity
    widths = [s.get("width", 3) for s in strokes]
    widths = [w for w in widths if w is not None]
    if widths:
        feats["db_width_unique"] = len(set(widths))
        feats["db_width_mean"] = float(np.mean(widths))

    return feats


def spatial_features(strokes: List[Dict], map_w: int = 651, map_h: int = 900) -> Dict[str, float]:
    """Spatial distribution features of the drawing."""
    draw_strokes = [s for s in strokes if s.get("mode", "draw") == "draw"]
    if not draw_strokes:
        return {}

    all_x, all_y = [], []
    for s in draw_strokes:
        xs, ys, _ = _point_arrays(s)
        all_x.extend(xs.tolist())
        all_y.extend(ys.tolist())

    if not all_x:
        return {}

    ax = np.array(all_x)
    ay = np.array(all_y)

    feats = {
        "ds_centroid_x": float(np.mean(ax)),
        "ds_centroid_y": float(np.mean(ay)),
        "ds_spread_x": float(np.std(ax, ddof=1)) if len(ax) > 1 else 0.0,
        "ds_spread_y": float(np.std(ay, ddof=1)) if len(ay) > 1 else 0.0,
        "ds_bbox_x1": float(np.min(ax)),
        "ds_bbox_y1": float(np.min(ay)),
        "ds_bbox_x2": float(np.max(ax)),
        "ds_bbox_y2": float(np.max(ay)),
    }

    bbox_w = feats["ds_bbox_x2"] - feats["ds_bbox_x1"]
    bbox_h = feats["ds_bbox_y2"] - feats["ds_bbox_y1"]
    feats["ds_bbox_area"] = bbox_w * bbox_h
    feats["ds_map_coverage"] = feats["ds_bbox_area"] / (map_w * map_h)

    # Convex hull area
    if len(ax) >= 3:
        try:
            from scipy.spatial import ConvexHull
            pts = np.column_stack([ax, ay])
            hull = ConvexHull(pts)
            feats["ds_convex_hull_area"] = float(hull.volume)
        except Exception:
            feats["ds_convex_hull_area"] = 0.0

    # Spatial entropy (10x10 grid)
    grid_n = 10
    gw = map_w / grid_n
    gh = map_h / grid_n
    cells = Counter()
    for x, y in zip(ax, ay):
        cx = min(int(x / gw), grid_n - 1)
        cy = min(int(y / gh), grid_n - 1)
        cells[(cx, cy)] += 1
    total = sum(cells.values())
    if total > 0:
        probs = np.array([c / total for c in cells.values()])
        feats["ds_spatial_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-12)))
        feats["ds_spatial_entropy_norm"] = feats["ds_spatial_entropy"] / np.log2(grid_n * grid_n)
        feats["ds_cells_occupied"] = len(cells)
        feats["ds_grid_coverage"] = len(cells) / (grid_n * grid_n)

    # Direction distribution of drawing movement
    all_dirs = []
    for s in draw_strokes:
        xs, ys, _ = _point_arrays(s)
        if len(xs) > 1:
            dx = np.diff(xs)
            dy = np.diff(ys)
            dirs = np.degrees(np.arctan2(dy, dx))
            all_dirs.extend(dirs.tolist())

    if all_dirs:
        dirs_arr = np.array(all_dirs)
        bins = np.arange(-180, 181, 45)
        hist, _ = np.histogram(dirs_arr, bins=bins)
        total_d = hist.sum()
        # Screen coords: y increases downward, so N/S are inverted vs math convention
        labels = ["W", "NW", "N", "NE", "E", "SE", "S", "SW"]
        for label, count in zip(labels, hist):
            feats[f"ds_dir_{label.lower()}"] = count / total_d if total_d > 0 else 0.0

        # Direction entropy
        if total_d > 0:
            probs_d = hist[hist > 0] / total_d
            feats["ds_direction_entropy"] = float(-np.sum(probs_d * np.log2(probs_d + 1e-12)))

    return feats


def progression_features(strokes: List[Dict], gt_json: Dict = None,
                         map_w: int = 651, map_h: int = 900) -> Dict[str, float]:
    """Drawing progression over time: coverage growth, route-order adherence."""
    draw_strokes = [s for s in strokes if s.get("mode", "draw") == "draw"]
    if not draw_strokes:
        return {}

    # Sort strokes by first point time
    def stroke_t(s):
        pts = s.get("points", [])
        for p in pts:
            t = _sf(p.get("t"))
            if t is not None:
                return t
        return 0
    sorted_strokes = sorted(draw_strokes, key=stroke_t)

    # Cumulative path length over strokes
    cum_lengths = []
    cum_len = 0
    for s in sorted_strokes:
        xs, ys, _ = _point_arrays(s)
        cum_len += _stroke_length(xs, ys)
        cum_lengths.append(cum_len)

    feats = {}
    if len(cum_lengths) > 1:
        # Length growth linearity (R^2 of cumulative length vs stroke index)
        x_idx = np.arange(len(cum_lengths))
        if np.std(cum_lengths) > 0:
            r = np.corrcoef(x_idx, cum_lengths)[0, 1]
            feats["dp_length_growth_r"] = float(r) if np.isfinite(r) else 0.0

    feats["dp_total_path_length"] = cum_len

    # Stroke-order vs GT route order (if GT available)
    if gt_json:
        gt_pts = []
        for s in gt_json.get("strokes", []):
            for p in s.get("points", s.get("polyline", [])):
                if isinstance(p, dict) and "x" in p:
                    gt_pts.append((p["x"], p["y"]))

        if gt_pts and len(sorted_strokes) >= 2:
            gt_arr = np.array(gt_pts)
            # For each stroke midpoint, find nearest GT point index
            stroke_gt_indices = []
            for s in sorted_strokes:
                xs, ys, _ = _point_arrays(s)
                if len(xs) == 0:
                    continue
                mid_x = float(np.mean(xs))
                mid_y = float(np.mean(ys))
                dists = np.sqrt((gt_arr[:, 0] - mid_x) ** 2 + (gt_arr[:, 1] - mid_y) ** 2)
                stroke_gt_indices.append(int(np.argmin(dists)))

            if len(stroke_gt_indices) >= 2:
                # Correlation between stroke order and GT route order
                from scipy.stats import kendalltau
                tau, p_val = kendalltau(range(len(stroke_gt_indices)), stroke_gt_indices)
                feats["dp_route_order_tau"] = float(tau) if np.isfinite(tau) else 0.0
                feats["dp_route_order_p"] = float(p_val) if np.isfinite(p_val) else 1.0

                # Is drawing monotonic along route?
                monotonic_pairs = sum(
                    1 for i in range(len(stroke_gt_indices) - 1)
                    if stroke_gt_indices[i + 1] > stroke_gt_indices[i])
                feats["dp_monotonic_pct"] = monotonic_pairs / (len(stroke_gt_indices) - 1)

    return feats


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def extract_drawing_features(strokes: List[Dict], trial: int = 0,
                             gt_json: Dict = None,
                             trial_duration_ms: float = 210000) -> Dict[str, Any]:
    """
    Extract all drawing behavior features for one trial.

    Args:
        strokes: list of stroke dicts with {points: [{x,y,t}], mode, width}
        trial: trial index
        gt_json: optional ground truth JSON for route-order comparison
        trial_duration_ms: trial duration for duty cycle calculation
    """
    result = {"trial": trial}

    if not strokes:
        result["drawing_n_strokes"] = 0
        return result

    result.update(temporal_features(strokes, trial_duration_ms))
    result.update(kinematic_features(strokes))
    result.update(behavioral_features(strokes))
    result.update(spatial_features(strokes))
    result.update(progression_features(strokes, gt_json))

    return result


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Extract drawing behavior features")
    ap.add_argument("--strokes", required=True, help="Path to strokes.json")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--gt", default=None, help="Ground truth JSON path")
    args = ap.parse_args()

    with open(args.strokes) as f:
        strokes = json.load(f)

    gt = None
    if args.gt:
        with open(args.gt) as f:
            gt = json.load(f)

    feats = extract_drawing_features(strokes, args.trial, gt)
    print(json.dumps(feats, indent=2, default=str))
