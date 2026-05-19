#!/usr/bin/env python3
"""
Comprehensive gaze feature extraction from preprocessed eye-tracker CSV.

Reads unified CSV from preprocess_eye.py, computes ~100 features per individual
and per dyad. Handles both Aurora (full event data) and SmartEye (indices only,
computes missing metrics from raw gaze).

Usage:
    from gaze_features import extract_gaze_features, extract_pair_gaze_features

Requirements:
    pip install numpy scipy pyrqa
"""

import csv
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.computation import RQAComputation
    from pyrqa.metric import EuclideanMetric
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.analysis_type import Cross
    HAS_RQA = True
except Exception:
    HAS_RQA = False

# ── Map coordinate normalization ──

AOI_CONFIG = {
    "director": {"map": {"x1": 252, "y1": 137, "x2": 889, "y2": 1017}},
    "matcher": {"map": {"x1": 267, "y1": 137, "x2": 904, "y2": 1017}},
}

MAP_W, MAP_H = 651, 900


def _sf(v, default=None):
    """Safe float conversion."""
    if v is None or v == "":
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


# ── CSV loading ──

def load_gaze_csv(path: str) -> List[Dict]:
    """Load preprocessed eye CSV into list of dicts with parsed numeric fields."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def filter_trial(rows: List[Dict], trial: str) -> List[Dict]:
    """Filter rows to a specific trial label."""
    return [r for r in rows if r.get("trial") == trial]


def _to_map_coords(gx, gy, role: str) -> Tuple[Optional[float], Optional[float]]:
    """Convert screen gaze to normalized map coordinates (0-651, 0-900)."""
    if gx is None or gy is None:
        return None, None
    aoi = AOI_CONFIG.get(role, AOI_CONFIG["director"])["map"]
    mx = (gx - aoi["x1"]) / (aoi["x2"] - aoi["x1"]) * MAP_W
    my = (gy - aoi["y1"]) / (aoi["y2"] - aoi["y1"]) * MAP_H
    return mx, my


# ── SmartEye gap filling ──

def _enrich_smarteye(rows: List[Dict]) -> List[Dict]:
    """Compute missing fixation/saccade metrics for SmartEye data."""
    if not rows or rows[0].get("source") != "smarteye":
        return rows

    # Group by fixation_idx to compute fixation x/y/duration
    fix_groups = defaultdict(list)
    for i, r in enumerate(rows):
        fidx = r.get("fixation_idx", "")
        if fidx and fidx != "" and fidx != "0":
            fix_groups[fidx].append(i)

    for fidx, indices in fix_groups.items():
        gx_vals = [_sf(rows[i]["gaze_x"]) for i in indices]
        gy_vals = [_sf(rows[i]["gaze_y"]) for i in indices]
        ts_vals = [_sf(rows[i]["t_unix_ms"]) for i in indices]
        gx_vals = [v for v in gx_vals if v is not None]
        gy_vals = [v for v in gy_vals if v is not None]
        ts_vals = [v for v in ts_vals if v is not None]

        fx = float(np.mean(gx_vals)) if gx_vals else None
        fy = float(np.mean(gy_vals)) if gy_vals else None
        fdur = (max(ts_vals) - min(ts_vals)) if len(ts_vals) > 1 else 0.0

        for i in indices:
            if fx is not None:
                rows[i]["fixation_x"] = fx
            if fy is not None:
                rows[i]["fixation_y"] = fy
            rows[i]["fixation_duration"] = fdur

    # Compute gaze velocity between consecutive samples
    for i in range(1, len(rows)):
        t0 = _sf(rows[i - 1]["t_unix_ms"])
        t1 = _sf(rows[i]["t_unix_ms"])
        gx0, gy0 = _sf(rows[i - 1]["gaze_x"]), _sf(rows[i - 1]["gaze_y"])
        gx1, gy1 = _sf(rows[i]["gaze_x"]), _sf(rows[i]["gaze_y"])
        if all(v is not None for v in [t0, t1, gx0, gy0, gx1, gy1]) and t1 > t0:
            dt = (t1 - t0) / 1000.0  # seconds
            dist = math.sqrt((gx1 - gx0) ** 2 + (gy1 - gy0) ** 2)
            rows[i]["gaze_velocity"] = dist / dt if dt > 0 else 0.0

    # Compute saccade metrics from fixation transitions
    sacc_groups = defaultdict(list)
    for i, r in enumerate(rows):
        sidx = r.get("saccade_idx", "")
        if sidx and sidx != "" and sidx != "0":
            sacc_groups[sidx].append(i)

    for sidx, indices in sacc_groups.items():
        gx_vals = [(_sf(rows[i]["gaze_x"]), _sf(rows[i]["gaze_y"])) for i in indices]
        gx_vals = [(x, y) for x, y in gx_vals if x is not None and y is not None]
        vel_vals = [_sf(rows[i].get("gaze_velocity")) for i in indices]
        vel_vals = [v for v in vel_vals if v is not None]

        if len(gx_vals) >= 2:
            amplitude = math.sqrt(
                (gx_vals[-1][0] - gx_vals[0][0]) ** 2 +
                (gx_vals[-1][1] - gx_vals[0][1]) ** 2)
            direction = math.degrees(math.atan2(
                gx_vals[-1][1] - gx_vals[0][1],
                gx_vals[-1][0] - gx_vals[0][0]))
        else:
            amplitude = 0.0
            direction = 0.0

        peak_vel = max(vel_vals) if vel_vals else 0.0

        for i in indices:
            rows[i]["saccade_amplitude"] = amplitude
            rows[i]["saccade_peak_velocity"] = peak_vel
            rows[i]["saccade_direction"] = direction

    return rows


def _detect_blinks_aurora(rows: List[Dict]) -> List[Dict]:
    """Detect blinks from Aurora eyelid opening data (no explicit blink column)."""
    if not rows or rows[0].get("source") != "aurora":
        return rows
    # Aurora has eyelid opening columns; low values = blink
    # Also fall back to pupil data loss when eyelid data unavailable
    for r in rows:
        if r.get("blink"):
            continue
        el = _sf(r.get("eyelid_left"))
        er = _sf(r.get("eyelid_right"))
        # Eyelid opening near zero = blink (threshold ~0.002mm)
        if el is not None and er is not None and el < 0.002 and er < 0.002:
            r["blink"] = "1"
            continue
        pl = _sf(r.get("pupil_left"))
        pr = _sf(r.get("pupil_right"))
        gx = _sf(r.get("gaze_x"))
        if (pl is None or pl <= 0) and (pr is None or pr <= 0) and gx is None:
            r["blink"] = "1"
    return rows


# ── Helper: extract typed arrays from rows ──

def _gaze_arrays(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps_ms, gaze_x, gaze_y) arrays for valid gaze samples."""
    ts, gx, gy = [], [], []
    for r in rows:
        t = _sf(r.get("t_unix_ms"))
        x = _sf(r.get("gaze_x"))
        y = _sf(r.get("gaze_y"))
        if t is not None and x is not None and y is not None:
            ts.append(t)
            gx.append(x)
            gy.append(y)
    return np.array(ts), np.array(gx), np.array(gy)


def _fixation_list(rows: List[Dict]) -> List[Dict]:
    """Extract unique fixation events with centroid and duration."""
    fix_map = defaultdict(lambda: {"xs": [], "ys": [], "ts": [], "durs": []})
    for r in rows:
        fidx = r.get("fixation_idx", "")
        if not fidx or fidx == "" or fidx == "0":
            continue
        fx = _sf(r.get("fixation_x")) or _sf(r.get("gaze_x"))
        fy = _sf(r.get("fixation_y")) or _sf(r.get("gaze_y"))
        ft = _sf(r.get("t_unix_ms"))
        fd = _sf(r.get("fixation_duration"))
        if fx is not None:
            fix_map[fidx]["xs"].append(fx)
        if fy is not None:
            fix_map[fidx]["ys"].append(fy)
        if ft is not None:
            fix_map[fidx]["ts"].append(ft)
        if fd is not None and fd not in fix_map[fidx]["durs"]:
            fix_map[fidx]["durs"].append(fd)

    fixations = []
    for fidx in sorted(fix_map.keys(), key=lambda x: min(fix_map[x]["ts"]) if fix_map[x]["ts"] else 0):
        d = fix_map[fidx]
        if not d["xs"] or not d["ts"]:
            continue
        dur = d["durs"][0] if d["durs"] else (max(d["ts"]) - min(d["ts"]))
        fixations.append({
            "idx": fidx,
            "x": float(np.mean(d["xs"])),
            "y": float(np.mean(d["ys"])),
            "t_start": min(d["ts"]),
            "t_end": max(d["ts"]),
            "duration": dur,
        })
    return fixations


def _saccade_list(rows: List[Dict]) -> List[Dict]:
    """Extract unique saccade events."""
    sacc_map = defaultdict(lambda: {"amps": [], "vels": [], "dirs": [], "ts": []})
    for r in rows:
        sidx = r.get("saccade_idx", "")
        if not sidx or sidx == "" or sidx == "0":
            continue
        sa = _sf(r.get("saccade_amplitude"))
        sv = _sf(r.get("saccade_peak_velocity"))
        sd = _sf(r.get("saccade_direction"))
        st = _sf(r.get("t_unix_ms"))
        if sa is not None:
            sacc_map[sidx]["amps"].append(sa)
        if sv is not None:
            sacc_map[sidx]["vels"].append(sv)
        if sd is not None:
            sacc_map[sidx]["dirs"].append(sd)
        if st is not None:
            sacc_map[sidx]["ts"].append(st)

    saccades = []
    for sidx in sorted(sacc_map.keys(), key=lambda x: min(sacc_map[x]["ts"]) if sacc_map[x]["ts"] else 0):
        d = sacc_map[sidx]
        saccades.append({
            "idx": sidx,
            "amplitude": d["amps"][0] if d["amps"] else 0.0,
            "peak_velocity": max(d["vels"]) if d["vels"] else 0.0,
            "direction": d["dirs"][0] if d["dirs"] else 0.0,
            "t_start": min(d["ts"]) if d["ts"] else 0,
        })
    return saccades


# ═══════════════════════════════════════════════════════════════════════
# PER-INDIVIDUAL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def fixation_features(rows: List[Dict]) -> Dict[str, float]:
    """Compute fixation metrics from gaze rows for one trial/participant."""
    fixations = _fixation_list(rows)
    if not fixations:
        return {}

    durations = np.array([f["duration"] for f in fixations])
    xs = np.array([f["x"] for f in fixations])
    ys = np.array([f["y"] for f in fixations])

    # Trial duration
    ts_all = [_sf(r.get("t_unix_ms")) for r in rows]
    ts_all = [t for t in ts_all if t is not None]
    trial_dur_s = (max(ts_all) - min(ts_all)) / 1000.0 if len(ts_all) > 1 else 1.0

    feats = {
        "fix_count": len(fixations),
        "fix_rate": len(fixations) / trial_dur_s if trial_dur_s > 0 else 0.0,
        "fix_dur_mean": float(np.mean(durations)),
        "fix_dur_median": float(np.median(durations)),
        "fix_dur_std": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
        "fix_dur_max": float(np.max(durations)),
        "fix_dur_min": float(np.min(durations)),
    }

    # Spatial dispersion
    feats["fix_dispersion_x"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    feats["fix_dispersion_y"] = float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0
    feats["fix_dispersion"] = math.sqrt(feats["fix_dispersion_x"] ** 2 + feats["fix_dispersion_y"] ** 2)

    # Convex hull area
    if len(xs) >= 3:
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack([xs, ys])
            hull = ConvexHull(points)
            feats["fix_convex_hull_area"] = float(hull.volume)  # 2D: volume = area
        except Exception:
            feats["fix_convex_hull_area"] = 0.0
    else:
        feats["fix_convex_hull_area"] = 0.0

    # Ambient/focal ratio (Velichkovsky)
    ambient = sum(1 for d in durations if d < 150)
    focal = sum(1 for d in durations if d > 300)
    feats["fix_ambient_count"] = ambient
    feats["fix_focal_count"] = focal
    feats["fix_ambient_focal_ratio"] = ambient / focal if focal > 0 else float("inf")

    # Fixation-to-saccade time ratio
    total_fix_time = float(np.sum(durations))
    feats["fix_total_time_ms"] = total_fix_time
    feats["fix_saccade_time_ratio"] = total_fix_time / (trial_dur_s * 1000) if trial_dur_s > 0 else 0.0

    return feats


def saccade_features(rows: List[Dict]) -> Dict[str, float]:
    """Compute saccade metrics."""
    saccades = _saccade_list(rows)
    if not saccades:
        return {}

    amps = np.array([s["amplitude"] for s in saccades])
    vels = np.array([s["peak_velocity"] for s in saccades])
    dirs_deg = np.array([s["direction"] for s in saccades])

    ts_all = [_sf(r.get("t_unix_ms")) for r in rows]
    ts_all = [t for t in ts_all if t is not None]
    trial_dur_s = (max(ts_all) - min(ts_all)) / 1000.0 if len(ts_all) > 1 else 1.0

    feats = {
        "sacc_count": len(saccades),
        "sacc_rate": len(saccades) / trial_dur_s if trial_dur_s > 0 else 0.0,
        "sacc_amp_mean": float(np.mean(amps)),
        "sacc_amp_std": float(np.std(amps, ddof=1)) if len(amps) > 1 else 0.0,
        "sacc_amp_max": float(np.max(amps)),
        "sacc_vel_mean": float(np.mean(vels)) if vels.size else 0.0,
        "sacc_vel_max": float(np.max(vels)) if vels.size else 0.0,
    }

    # Main sequence slope (amplitude vs peak velocity)
    if len(amps) > 2 and np.std(amps) > 0:
        slope = float(np.polyfit(amps, vels, 1)[0])
        feats["sacc_main_sequence_slope"] = slope

    # Direction distribution (8 bins: N, NE, E, SE, S, SW, W, NW)
    bins = np.arange(-180, 181, 45)
    hist, _ = np.histogram(dirs_deg, bins=bins)
    total = hist.sum()
    dir_labels = ["W", "NW", "N", "NE", "E", "SE", "S", "SW"]
    for label, count in zip(dir_labels, hist):
        feats[f"sacc_dir_{label.lower()}"] = count / total if total > 0 else 0.0

    # Regressive saccades (going backward — leftward or upward on map)
    regressive = sum(1 for d in dirs_deg if 90 < abs(d) <= 180)
    feats["sacc_regressive_rate"] = regressive / len(dirs_deg) if dirs_deg.size else 0.0

    return feats


def pupil_features(rows: List[Dict], baseline_rows: List[Dict] = None) -> Dict[str, float]:
    """Compute pupil metrics. Optional baseline_rows for correction."""
    pl_vals = [_sf(r.get("pupil_left")) for r in rows]
    pr_vals = [_sf(r.get("pupil_right")) for r in rows]
    pl_vals = [v for v in pl_vals if v is not None and v > 0]
    pr_vals = [v for v in pr_vals if v is not None and v > 0]

    if not pl_vals and not pr_vals:
        return {}

    # Average of both eyes where available
    pupil_mean_arr = []
    for r in rows:
        pl = _sf(r.get("pupil_left"))
        pr = _sf(r.get("pupil_right"))
        vals = [v for v in [pl, pr] if v is not None and v > 0]
        if vals:
            pupil_mean_arr.append(float(np.mean(vals)))

    if not pupil_mean_arr:
        return {}

    pupil = np.array(pupil_mean_arr)

    feats = {
        "pupil_mean": float(np.mean(pupil)),
        "pupil_std": float(np.std(pupil, ddof=1)) if len(pupil) > 1 else 0.0,
        "pupil_min": float(np.min(pupil)),
        "pupil_max": float(np.max(pupil)),
        "pupil_range": float(np.max(pupil) - np.min(pupil)),
        "pupil_median": float(np.median(pupil)),
        "pupil_data_pct": len(pupil_mean_arr) / max(len(rows), 1),
    }

    # Left/right separately
    if pl_vals:
        feats["pupil_left_mean"] = float(np.mean(pl_vals))
    if pr_vals:
        feats["pupil_right_mean"] = float(np.mean(pr_vals))
    if pl_vals and pr_vals:
        feats["pupil_asymmetry"] = abs(float(np.mean(pl_vals)) - float(np.mean(pr_vals)))

    # Baseline correction
    if baseline_rows:
        bl_pupil = []
        for r in baseline_rows:
            pl = _sf(r.get("pupil_left"))
            pr = _sf(r.get("pupil_right"))
            vals = [v for v in [pl, pr] if v is not None and v > 0]
            if vals:
                bl_pupil.append(float(np.mean(vals)))
        if bl_pupil:
            bl_mean = float(np.mean(bl_pupil))
            feats["pupil_baseline_mean"] = bl_mean
            corrected = pupil - bl_mean
            feats["pupil_corrected_mean"] = float(np.mean(corrected))
            feats["pupil_corrected_max"] = float(np.max(corrected))  # TEPR
            feats["pupil_tepr"] = float(np.max(corrected))

    # Dilation rate (derivative)
    ts = [_sf(r.get("t_unix_ms")) for r in rows]
    ts_valid = []
    pupil_valid = []
    for i, r in enumerate(rows):
        t = _sf(r.get("t_unix_ms"))
        pl = _sf(r.get("pupil_left"))
        pr = _sf(r.get("pupil_right"))
        vals = [v for v in [pl, pr] if v is not None and v > 0]
        if t is not None and vals:
            ts_valid.append(t)
            pupil_valid.append(float(np.mean(vals)))

    if len(ts_valid) > 2:
        ts_arr = np.array(ts_valid)
        pup_arr = np.array(pupil_valid)
        dt = np.diff(ts_arr) / 1000.0  # seconds
        dp = np.diff(pup_arr)
        rate = dp / np.where(dt > 0, dt, 1e-6)
        feats["pupil_dilation_rate_mean"] = float(np.mean(rate))
        feats["pupil_dilation_rate_std"] = float(np.std(rate))

    # Low-frequency fluctuations (0.04-0.15 Hz)
    if len(pupil_valid) > 30:
        try:
            from scipy.signal import welch
            # Estimate sampling rate
            ts_arr = np.array(ts_valid)
            fs = 1000.0 / np.median(np.diff(ts_arr)) if len(ts_arr) > 1 else 60.0
            fs = min(max(fs, 10), 120)  # clamp
            freqs, psd = welch(np.array(pupil_valid) - np.mean(pupil_valid),
                               fs=fs, nperseg=min(len(pupil_valid), int(fs * 30)))
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            if lf_mask.any():
                df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                feats["pupil_lf_power"] = float(np.trapz(psd[lf_mask], dx=df))
        except Exception:
            pass

    return feats


def blink_features(rows: List[Dict]) -> Dict[str, float]:
    """Compute blink metrics."""
    ts_all = [_sf(r.get("t_unix_ms")) for r in rows]
    ts_all = [t for t in ts_all if t is not None]
    trial_dur_s = (max(ts_all) - min(ts_all)) / 1000.0 if len(ts_all) > 1 else 1.0

    # Detect blink episodes (consecutive blink=1 samples)
    blink_episodes = []
    in_blink = False
    blink_start = None

    for r in rows:
        is_blink = str(r.get("blink", "")).strip() in ("1", "True", "true")
        t = _sf(r.get("t_unix_ms"))
        if is_blink and not in_blink:
            in_blink = True
            blink_start = t
        elif not is_blink and in_blink:
            in_blink = False
            if blink_start is not None and t is not None:
                blink_episodes.append(t - blink_start)
            blink_start = None

    # Close final blink if still open at end of data
    if in_blink and blink_start is not None:
        last_t = _sf(rows[-1].get("t_unix_ms"))
        if last_t is not None:
            blink_episodes.append(last_t - blink_start)

    blink_count = len(blink_episodes)
    feats = {
        "blink_count": blink_count,
        "blink_rate_per_min": blink_count / (trial_dur_s / 60.0) if trial_dur_s > 0 else 0.0,
    }

    if blink_episodes:
        durs = np.array(blink_episodes)
        feats["blink_dur_mean"] = float(np.mean(durs))
        feats["blink_dur_std"] = float(np.std(durs)) if len(durs) > 1 else 0.0
        feats["blink_dur_max"] = float(np.max(durs))

    return feats


def scanpath_features(rows: List[Dict]) -> Dict[str, float]:
    """Compute scanpath metrics: length, entropy, NNI."""
    fixations = _fixation_list(rows)
    if len(fixations) < 2:
        return {}

    xs = np.array([f["x"] for f in fixations])
    ys = np.array([f["y"] for f in fixations])

    # Scanpath length (total Euclidean distance)
    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    scan_length = float(np.sum(diffs))
    scan_dur = fixations[-1]["t_end"] - fixations[0]["t_start"]

    feats = {
        "scan_length_px": scan_length,
        "scan_duration_ms": scan_dur,
        "scan_velocity_px_s": scan_length / (scan_dur / 1000.0) if scan_dur > 0 else 0.0,
    }

    # Stationary Gaze Entropy (SGE) — spatial grid over fixed map bounds
    n_bins = 10  # 10x10 grid
    x_edges = np.linspace(0, 651, n_bins + 1)
    y_edges = np.linspace(0, 900, n_bins + 1)
    cells = Counter()
    for x, y in zip(xs, ys):
        cx = min(int(np.searchsorted(x_edges, x, side="right") - 1), n_bins - 1)
        cy = min(int(np.searchsorted(y_edges, y, side="right") - 1), n_bins - 1)
        cx = max(cx, 0)
        cy = max(cy, 0)
        cells[(cx, cy)] += 1
    total = sum(cells.values())
    probs = np.array([c / total for c in cells.values()])
    sge = float(-np.sum(probs * np.log2(probs + 1e-12)))
    feats["scan_entropy_sge"] = sge
    feats["scan_entropy_max"] = float(np.log2(n_bins * n_bins))
    feats["scan_entropy_norm"] = sge / feats["scan_entropy_max"] if feats["scan_entropy_max"] > 0 else 0.0

    # Nearest Neighbor Index
    if len(xs) > 2:
        from scipy.spatial.distance import cdist
        points = np.column_stack([xs, ys])
        D = cdist(points, points)
        np.fill_diagonal(D, np.inf)
        nn_dists = D.min(axis=1)
        observed_mean = float(np.mean(nn_dists))
        # Expected under CSR (complete spatial randomness)
        area = feats.get("fix_convex_hull_area", (max(xs) - min(xs)) * (max(ys) - min(ys)))
        if area <= 0:
            area = 1.0
        expected_mean = 0.5 * math.sqrt(area / len(xs))
        feats["scan_nni"] = observed_mean / expected_mean if expected_mean > 0 else 1.0

    return feats


def aoi_features(rows: List[Dict]) -> Dict[str, float]:
    """Compute AOI-based metrics: dwell time, fixation count, transitions, coverage."""
    if not rows:
        return {}

    ts_all = [_sf(r.get("t_unix_ms")) for r in rows]
    ts_all = [t for t in ts_all if t is not None]
    trial_dur_ms = (max(ts_all) - min(ts_all)) if len(ts_all) > 1 else 1.0

    # Estimate sample interval
    dts = np.diff(sorted(ts_all))
    sample_interval_ms = float(np.median(dts)) if len(dts) > 0 else 16.67

    aoi_time = Counter()
    aoi_fix_count = Counter()
    aoi_first_fixation = {}
    aoi_revisit = Counter()
    prev_aoi = None
    visited_aois = set()

    # AOI transitions
    transitions = Counter()

    fixations = _fixation_list(rows)
    for f in fixations:
        # Classify fixation centroid into AOI
        aoi = "other"
        for r in rows:
            fidx = r.get("fixation_idx", "")
            if fidx == f["idx"]:
                aoi = r.get("aoi", "other")
                break

        aoi_fix_count[aoi] += 1
        aoi_time[aoi] += f["duration"]

        if aoi not in aoi_first_fixation:
            aoi_first_fixation[aoi] = f["t_start"]

        if aoi in visited_aois and aoi != prev_aoi:
            aoi_revisit[aoi] += 1
        visited_aois.add(aoi)

        if prev_aoi is not None and prev_aoi != aoi:
            transitions[(prev_aoi, aoi)] += 1
        prev_aoi = aoi

    feats = {}
    all_aois = ["map", "timer", "toolbar", "other", "missing"]

    for aoi_name in all_aois:
        feats[f"aoi_{aoi_name}_dwell_ms"] = aoi_time.get(aoi_name, 0.0)
        feats[f"aoi_{aoi_name}_dwell_pct"] = aoi_time.get(aoi_name, 0.0) / trial_dur_ms if trial_dur_ms > 0 else 0.0
        feats[f"aoi_{aoi_name}_fix_count"] = aoi_fix_count.get(aoi_name, 0)
        feats[f"aoi_{aoi_name}_revisits"] = aoi_revisit.get(aoi_name, 0)
        ft = aoi_first_fixation.get(aoi_name)
        t_start = min(ts_all) if ts_all else 0
        feats[f"aoi_{aoi_name}_ttff_ms"] = (ft - t_start) if ft is not None else ""

    # Coverage: proportion of non-missing AOIs visited
    content_aois = {"map", "timer", "toolbar"}
    feats["aoi_coverage"] = len(visited_aois & content_aois) / len(content_aois)

    # Gaze Transition Entropy (GTE) — conditional entropy
    # H = -sum_i p(i) * sum_j p(j|i) * log2(p(j|i))
    total_trans = sum(transitions.values())
    if total_trans > 0:
        # Build row sums for conditional probabilities
        row_sums = {}
        for (src, dst), count in transitions.items():
            row_sums[src] = row_sums.get(src, 0) + count
        gte = 0.0
        for (src, dst), count in transitions.items():
            p_joint = count / total_trans
            p_cond = count / row_sums[src]
            if p_cond > 0:
                gte -= p_joint * np.log2(p_cond + 1e-12)
        feats["aoi_transition_entropy"] = float(gte)
    else:
        feats["aoi_transition_entropy"] = 0.0

    # Data quality: gaze loss rate
    total_samples = len(rows)
    missing_samples = sum(1 for r in rows if r.get("aoi") == "missing" or _sf(r.get("gaze_x")) is None)
    feats["gaze_loss_rate"] = missing_samples / total_samples if total_samples > 0 else 1.0

    return feats


def gaze_rqa_features(rows: List[Dict]) -> Dict[str, float]:
    """Auto-RQA on gaze position time series."""
    if not HAS_RQA:
        return {}

    ts, gx, gy = _gaze_arrays(rows)
    if len(gx) < 20:
        return {}

    # Downsample to ~4Hz for RQA (every ~15th sample at 60Hz)
    step = max(1, len(gx) // 200)
    gx_ds = gx[::step]

    std_gx = float(np.std(gx_ds, ddof=1))
    threshold = max(0.1 * std_gx, 1.0) if std_gx > 0 else 1.0

    try:
        ts_rqa = TimeSeries(gx_ds.tolist(), embedding_dimension=2, time_delay=1)
        settings = Settings(ts_rqa, neighbourhood=FixedRadius(threshold),
                            similarity_measure=EuclideanMetric)
        comp = RQAComputation.create(settings)
        res = comp.run()
        sf = lambda v: float(v) if np.isfinite(float(v)) else 0.0
        return {
            "gaze_rqa_rr": sf(res.recurrence_rate),
            "gaze_rqa_det": sf(res.determinism),
            "gaze_rqa_mean_diag": sf(res.average_diagonal_line),
            "gaze_rqa_max_diag": sf(res.longest_diagonal_line),
            "gaze_rqa_entr_diag": sf(res.entropy_diagonal_lines),
            "gaze_rqa_lam": sf(res.laminarity),
            "gaze_rqa_tt": sf(res.trapping_time),
        }
    except Exception:
        return {}


def data_quality_features(rows: List[Dict]) -> Dict[str, float]:
    """Data quality and head movement features."""
    feats = {}

    total = len(rows)
    if total == 0:
        return {"dq_total_samples": 0}

    feats["dq_total_samples"] = total

    # Gaze data loss
    missing = sum(1 for r in rows if _sf(r.get("gaze_x")) is None)
    feats["dq_gaze_loss_pct"] = missing / total

    # Pupil data availability
    pupil_valid = sum(1 for r in rows if _sf(r.get("pupil_left")) is not None or _sf(r.get("pupil_right")) is not None)
    feats["dq_pupil_valid_pct"] = pupil_valid / total

    # Head movement magnitude
    pitch_vals = [_sf(r.get("head_pitch")) for r in rows]
    yaw_vals = [_sf(r.get("head_yaw")) for r in rows]
    roll_vals = [_sf(r.get("head_roll")) for r in rows]
    pitch_vals = [v for v in pitch_vals if v is not None]
    yaw_vals = [v for v in yaw_vals if v is not None]
    roll_vals = [v for v in roll_vals if v is not None]

    if pitch_vals:
        feats["dq_head_pitch_sd"] = float(np.std(pitch_vals))
    if yaw_vals:
        feats["dq_head_yaw_sd"] = float(np.std(yaw_vals))
    if roll_vals:
        feats["dq_head_roll_sd"] = float(np.std(roll_vals))

    return feats


# ═══════════════════════════════════════════════════════════════════════
# CROSS-PARTICIPANT FEATURES
# ═══════════════════════════════════════════════════════════════════════

def _map_gaze_arrays(rows: List[Dict], role: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get map-normalized gaze arrays for cross-participant comparison."""
    ts, mx, my = [], [], []
    for r in rows:
        t = _sf(r.get("t_unix_ms"))
        gx = _sf(r.get("gaze_x"))
        gy = _sf(r.get("gaze_y"))
        if t is None or gx is None or gy is None:
            continue
        nmx, nmy = _to_map_coords(gx, gy, role)
        if nmx is not None and 0 <= nmx <= MAP_W and 0 <= nmy <= MAP_H:
            ts.append(t)
            mx.append(nmx)
            my.append(nmy)
    return np.array(ts), np.array(mx), np.array(my)


def gaze_convergence_features(rows_d: List[Dict], rows_m: List[Dict]) -> Dict[str, float]:
    """Spatial gaze convergence between Director and Matcher in map coordinates."""
    ts_d, mx_d, my_d = _map_gaze_arrays(rows_d, "director")
    ts_m, mx_m, my_m = _map_gaze_arrays(rows_m, "matcher")

    if len(ts_d) < 10 or len(ts_m) < 10:
        return {}

    # Align to common time grid (resample to 10 Hz)
    t_start = max(ts_d[0], ts_m[0])
    t_end = min(ts_d[-1], ts_m[-1])
    if t_end <= t_start:
        return {}

    step_ms = 100  # 10 Hz
    t_grid = np.arange(t_start, t_end, step_ms)
    if len(t_grid) < 10:
        return {}

    # Interpolate both to grid
    dx = np.interp(t_grid, ts_d, mx_d)
    dy = np.interp(t_grid, ts_d, my_d)
    mx_i = np.interp(t_grid, ts_m, mx_m)
    my_i = np.interp(t_grid, ts_m, my_m)

    distances = np.sqrt((dx - mx_i) ** 2 + (dy - my_i) ** 2)

    feats = {
        "gaze_conv_mean_dist": float(np.mean(distances)),
        "gaze_conv_median_dist": float(np.median(distances)),
        "gaze_conv_std_dist": float(np.std(distances)),
        "gaze_conv_min_dist": float(np.min(distances)),
        "gaze_conv_max_dist": float(np.max(distances)),
    }

    # Joint AOI fixation: proportion of time both within 50px of each other
    for threshold in [50, 100, 150]:
        pct = float(np.mean(distances < threshold))
        feats[f"gaze_conv_pct_within_{threshold}px"] = pct

    # Convergence over time (trend): does convergence improve during trial?
    if len(distances) > 2:
        slope = float(np.polyfit(range(len(distances)), distances, 1)[0])
        feats["gaze_conv_trend"] = slope  # negative = improving

    return feats


def gaze_crqa_features(rows_d: List[Dict], rows_m: List[Dict]) -> Dict[str, float]:
    """Cross-recurrence quantification analysis on map-normalized gaze."""
    if not HAS_RQA:
        return {}

    ts_d, mx_d, my_d = _map_gaze_arrays(rows_d, "director")
    ts_m, mx_m, my_m = _map_gaze_arrays(rows_m, "matcher")

    if len(mx_d) < 20 or len(mx_m) < 20:
        return {}

    # Time-align both gaze series to common 10Hz grid before CRQA
    t_start = max(ts_d[0], ts_m[0])
    t_end = min(ts_d[-1], ts_m[-1])
    if t_end <= t_start:
        return {}
    grid = np.arange(t_start, t_end, 100)  # 10 Hz = 100ms intervals
    if len(grid) < 20:
        return {}
    dx = np.interp(grid, ts_d, mx_d)
    mx_s = np.interp(grid, ts_m, mx_m)

    # Downsample to ~4 Hz for tractable RQA
    step = max(1, len(dx) // 200)
    dx = dx[::step]
    mx_s = mx_s[::step]

    std_both = float(np.std(np.concatenate([dx, mx_s]), ddof=1))
    threshold = max(0.1 * std_both, 1.0) if std_both > 0 else 1.0

    try:
        ts_d_rqa = TimeSeries(dx.tolist(), embedding_dimension=2, time_delay=1)
        ts_m_rqa = TimeSeries(mx_s.tolist(), embedding_dimension=2, time_delay=1)
        settings = Settings((ts_d_rqa, ts_m_rqa),
                            analysis_type=Cross,
                            neighbourhood=FixedRadius(threshold),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=0)
        comp = RQAComputation.create(settings)
        res = comp.run()
        sf = lambda v: float(v) if np.isfinite(float(v)) else 0.0
        return {
            "gaze_crqa_rr": sf(res.recurrence_rate),
            "gaze_crqa_det": sf(res.determinism),
            "gaze_crqa_mean_diag": sf(res.average_diagonal_line),
            "gaze_crqa_max_diag": sf(res.longest_diagonal_line),
            "gaze_crqa_entr_diag": sf(res.entropy_diagonal_lines),
            "gaze_crqa_lam": sf(res.laminarity),
            "gaze_crqa_tt": sf(res.trapping_time),
        }
    except Exception:
        return {}


def gaze_coupling_lag(rows_d: List[Dict], rows_m: List[Dict],
                      max_lag_ms: int = 5000) -> Dict[str, float]:
    """Compute gaze coupling lag via windowed cross-correlation in map space."""
    ts_d, mx_d, _ = _map_gaze_arrays(rows_d, "director")
    ts_m, mx_m, _ = _map_gaze_arrays(rows_m, "matcher")

    if len(mx_d) < 20 or len(mx_m) < 20:
        return {}

    # Align to common 10 Hz grid
    t_start = max(ts_d[0], ts_m[0])
    t_end = min(ts_d[-1], ts_m[-1])
    if t_end - t_start < 5000:
        return {}

    step_ms = 100
    t_grid = np.arange(t_start, t_end, step_ms)
    dx = np.interp(t_grid, ts_d, mx_d)
    mx_i = np.interp(t_grid, ts_m, mx_m)

    max_lag_samples = max_lag_ms // step_ms

    # Compute cross-correlation at different lags
    best_r = -2.0
    best_lag = 0
    lags = range(-max_lag_samples, max_lag_samples + 1)
    corrs = {}

    for lag in lags:
        if lag >= 0:
            d_seg = dx[:len(dx) - lag] if lag > 0 else dx
            m_seg = mx_i[lag:] if lag > 0 else mx_i
        else:
            d_seg = dx[-lag:]
            m_seg = mx_i[:len(mx_i) + lag]

        n = min(len(d_seg), len(m_seg))
        if n < 10:
            continue
        d_seg, m_seg = d_seg[:n], m_seg[:n]

        if np.std(d_seg) < 1e-6 or np.std(m_seg) < 1e-6:
            continue

        r = float(np.corrcoef(d_seg, m_seg)[0, 1])
        if np.isfinite(r):
            corrs[lag] = r
            if r > best_r:
                best_r = r
                best_lag = lag

    if not corrs:
        return {}

    return {
        "gaze_coupling_peak_lag_ms": best_lag * step_ms,
        "gaze_coupling_peak_r": best_r,
        "gaze_coupling_lag0_r": corrs.get(0, 0.0),
        "gaze_coupling_leader": "director" if best_lag > 0 else ("matcher" if best_lag < 0 else "sync"),
    }


# ═══════════════════════════════════════════════════════════════════════
# GAZE-SPEECH ALIGNMENT FEATURES
# ═══════════════════════════════════════════════════════════════════════

def gaze_speech_alignment(rows: List[Dict], words: List[Dict], role: str,
                          trial_start_ms: float = 0) -> Dict[str, float]:
    """
    Gaze-speech alignment for Director: where gaze is when speaking.

    Args:
        rows: gaze rows for this participant
        words: list of {word, start, end} from Whisper (seconds relative to audio start)
        trial_start_ms: Unix ms of audio recording start for time alignment
    """
    if not words or not rows:
        return {}

    ts, gx, gy = _gaze_arrays(rows)
    if len(ts) < 10:
        return {}

    # Gaze velocity during speech vs pauses
    speech_gaze_vels = []
    pause_gaze_vels = []

    for i, w in enumerate(words):
        word_start_ms = trial_start_ms + w.get("start", 0) * 1000
        word_end_ms = trial_start_ms + w.get("end", 0) * 1000

        # Find gaze samples during this word
        mask = (ts >= word_start_ms) & (ts <= word_end_ms)
        if mask.any():
            gx_w = gx[mask]
            gy_w = gy[mask]
            if len(gx_w) > 1:
                dists = np.sqrt(np.diff(gx_w) ** 2 + np.diff(gy_w) ** 2)
                speech_gaze_vels.extend(dists.tolist())

        # Check for pause before next word
        if i < len(words) - 1:
            pause_start = word_end_ms
            pause_end = trial_start_ms + words[i + 1].get("start", 0) * 1000
            if pause_end - pause_start > 300:  # >300ms pause
                mask_p = (ts >= pause_start) & (ts <= pause_end)
                if mask_p.any():
                    gx_p = gx[mask_p]
                    gy_p = gy[mask_p]
                    if len(gx_p) > 1:
                        dists_p = np.sqrt(np.diff(gx_p) ** 2 + np.diff(gy_p) ** 2)
                        pause_gaze_vels.extend(dists_p.tolist())

    feats = {}
    if speech_gaze_vels:
        feats["gs_speech_gaze_vel_mean"] = float(np.mean(speech_gaze_vels))
    if pause_gaze_vels:
        feats["gs_pause_gaze_vel_mean"] = float(np.mean(pause_gaze_vels))
    if speech_gaze_vels and pause_gaze_vels:
        feats["gs_speech_vs_pause_vel_ratio"] = (
            float(np.mean(speech_gaze_vels)) / float(np.mean(pause_gaze_vels))
            if np.mean(pause_gaze_vels) > 0 else 0.0)

    # Proportion of speech time with gaze on map
    speech_on_map = 0
    speech_total = 0
    for w in words:
        ws = trial_start_ms + w.get("start", 0) * 1000
        we = trial_start_ms + w.get("end", 0) * 1000
        for r in rows:
            t = _sf(r.get("t_unix_ms"))
            if t is not None and ws <= t <= we:
                speech_total += 1
                if r.get("aoi") == "map":
                    speech_on_map += 1

    feats["gs_speech_on_map_pct"] = speech_on_map / speech_total if speech_total > 0 else 0.0

    return feats


# ═══════════════════════════════════════════════════════════════════════
# GAZE-DRAWING ALIGNMENT FEATURES (Matcher only)
# ═══════════════════════════════════════════════════════════════════════

def gaze_drawing_alignment(rows: List[Dict], strokes: List[Dict],
                           role: str = "matcher") -> Dict[str, float]:
    """
    Gaze-drawing alignment for Matcher.

    Args:
        rows: gaze rows for matcher
        strokes: list of stroke dicts from strokes.json with points [{x,y,t}]
    """
    if not rows or not strokes:
        return {}

    ts, gx, gy = _gaze_arrays(rows)
    if len(ts) < 10:
        return {}

    aoi = AOI_CONFIG.get(role, AOI_CONFIG["matcher"])["map"]

    # Gaze-cursor distance during drawing
    gaze_cursor_dists = []
    gaze_at_stroke_onset = []

    for s in strokes:
        pts = s.get("points", [])
        if not pts:
            continue

        for p in pts:
            pt = _sf(p.get("t"))
            px = _sf(p.get("x"))
            py = _sf(p.get("y"))
            if pt is None or px is None or py is None:
                continue

            # Find nearest gaze sample
            idx = np.argmin(np.abs(ts - pt))
            if abs(ts[idx] - pt) < 100:  # within 100ms
                # Convert stroke coords (map space) to screen space for comparison
                screen_x = px / MAP_W * (aoi["x2"] - aoi["x1"]) + aoi["x1"]
                screen_y = py / MAP_H * (aoi["y2"] - aoi["y1"]) + aoi["y1"]
                dist = math.sqrt((gx[idx] - screen_x) ** 2 + (gy[idx] - screen_y) ** 2)
                gaze_cursor_dists.append(dist)

        # Gaze at stroke onset
        first_pt = pts[0]
        ft = _sf(first_pt.get("t"))
        if ft is not None:
            idx = np.argmin(np.abs(ts - ft))
            if abs(ts[idx] - ft) < 100:
                fx = _sf(first_pt.get("x"))
                fy = _sf(first_pt.get("y"))
                if fx is not None and fy is not None:
                    screen_x = fx / MAP_W * (aoi["x2"] - aoi["x1"]) + aoi["x1"]
                    screen_y = fy / MAP_H * (aoi["y2"] - aoi["y1"]) + aoi["y1"]
                    dist = math.sqrt((gx[idx] - screen_x) ** 2 + (gy[idx] - screen_y) ** 2)
                    gaze_at_stroke_onset.append(dist)

    feats = {}
    if gaze_cursor_dists:
        dists = np.array(gaze_cursor_dists)
        feats["gd_cursor_dist_mean"] = float(np.mean(dists))
        feats["gd_cursor_dist_median"] = float(np.median(dists))
        feats["gd_cursor_dist_std"] = float(np.std(dists))

    if gaze_at_stroke_onset:
        onset_dists = np.array(gaze_at_stroke_onset)
        feats["gd_onset_dist_mean"] = float(np.mean(onset_dists))
        feats["gd_onset_on_target_pct"] = float(np.mean(onset_dists < 50))  # within 50px

    # Gaze-on-own-drawing: fixation time on regions with existing strokes
    # Build list of (stroke_end_time, drawn_points) for all strokes first,
    # then iterate gaze rows ONCE to avoid double-counting.
    stroke_timeline = []  # list of (t_end, [(x, y), ...])
    for s in strokes:
        pts = s.get("points", [])
        if not pts:
            continue
        stroke_t_max = max((_sf(p.get("t")) or 0) for p in pts)
        spts = []
        for p in pts:
            px = _sf(p.get("x"))
            py = _sf(p.get("y"))
            if px is not None and py is not None:
                spts.append((px, py))
        stroke_timeline.append((stroke_t_max, spts))
    # Sort by completion time
    stroke_timeline.sort(key=lambda x: x[0])

    gaze_on_drawing = 0
    gaze_total_map = 0

    for r in rows:
        t = _sf(r.get("t_unix_ms"))
        g_x = _sf(r.get("gaze_x"))
        g_y = _sf(r.get("gaze_y"))
        if t is None or g_x is None or g_y is None or r.get("aoi") != "map":
            continue
        # Collect all drawn points from strokes completed before this gaze sample
        gmx, gmy = _to_map_coords(g_x, g_y, role)
        if gmx is None:
            continue
        # Check if any completed stroke has a point near gaze
        near_drawing = False
        for s_t_end, spts in stroke_timeline:
            if s_t_end >= t:
                break  # this stroke (and all after) not yet completed
            for dp in spts:
                if math.sqrt((gmx - dp[0]) ** 2 + (gmy - dp[1]) ** 2) < 30:
                    near_drawing = True
                    break
            if near_drawing:
                break
        # Only count gaze samples that occur after at least one stroke is done
        if stroke_timeline and t > stroke_timeline[0][0]:
            gaze_total_map += 1
            if near_drawing:
                gaze_on_drawing += 1

    if gaze_total_map > 0:
        feats["gd_gaze_on_drawing_pct"] = gaze_on_drawing / gaze_total_map

    return feats


# ═══════════════════════════════════════════════════════════════════════
# GAZE-ROUTE ALIGNMENT FEATURES
# ═══════════════════════════════════════════════════════════════════════

def gaze_route_alignment(rows: List[Dict], gt_json: Dict,
                         role: str) -> Dict[str, float]:
    """
    Compute gaze-route alignment features.

    Args:
        rows: gaze rows
        gt_json: ground truth JSON with route strokes
        role: director or matcher
    """
    if not rows or not gt_json:
        return {}

    # Extract GT route points in map coordinates
    route_pts = []
    for s in gt_json.get("strokes", []):
        for p in s.get("points", s.get("polyline", [])):
            if isinstance(p, dict) and "x" in p:
                route_pts.append((p["x"], p["y"]))

    if len(route_pts) < 2:
        return {}

    route_arr = np.array(route_pts)

    # Get map-normalized gaze
    ts, mx, my = _map_gaze_arrays(rows, role)
    if len(mx) < 10:
        return {}

    # Distance from each gaze point to nearest route point
    from scipy.spatial.distance import cdist
    gaze_pts = np.column_stack([mx, my])
    D = cdist(gaze_pts, route_arr)
    min_dists = D.min(axis=1)

    feats = {
        "gr_route_dist_mean": float(np.mean(min_dists)),
        "gr_route_dist_median": float(np.median(min_dists)),
        "gr_route_dist_std": float(np.std(min_dists)),
        "gr_on_route_pct": float(np.mean(min_dists < 30)),  # within 30px of route
    }

    # Route progress: map gaze to nearest route point index → progress 0-100%
    nearest_idx = D.argmin(axis=1)
    progress = nearest_idx / (len(route_pts) - 1) * 100  # 0-100%

    # Correlate progress with time
    if len(progress) > 2 and np.std(progress) > 0:
        time_norm = np.arange(len(progress)) / len(progress)
        corr = float(np.corrcoef(time_norm, progress)[0, 1])
        feats["gr_progress_correlation"] = corr if np.isfinite(corr) else 0.0
        feats["gr_progress_monotonic"] = 1.0 if corr > 0.7 else 0.0

    return feats


# ═══════════════════════════════════════════════════════════════════════
# MULTIMODAL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def pupil_hr_correlation(rows: List[Dict], hr_data: List[Dict]) -> Dict[str, float]:
    """Correlate pupil diameter with HR time series."""
    if not rows or not hr_data:
        return {}

    # Build pupil time series
    pupil_ts = []
    for r in rows:
        t = _sf(r.get("t_unix_ms"))
        pl = _sf(r.get("pupil_left"))
        pr = _sf(r.get("pupil_right"))
        vals = [v for v in [pl, pr] if v is not None and v > 0]
        if t is not None and vals:
            pupil_ts.append((t, float(np.mean(vals))))

    # Build HR time series
    hr_ts = []
    for r in hr_data:
        t = _sf(r.get("t")) or _sf(r.get("t_unix_ms"))
        bpm = _sf(r.get("bpm"))
        if t is not None and bpm is not None:
            hr_ts.append((t, bpm))

    if len(pupil_ts) < 10 or len(hr_ts) < 5:
        return {}

    # Align to common time grid (1 Hz)
    p_ts = np.array([p[0] for p in pupil_ts])
    p_vals = np.array([p[1] for p in pupil_ts])
    h_ts = np.array([h[0] for h in hr_ts])
    h_vals = np.array([h[1] for h in hr_ts])

    t_start = max(p_ts[0], h_ts[0])
    t_end = min(p_ts[-1], h_ts[-1])
    if t_end - t_start < 5000:
        return {}

    t_grid = np.arange(t_start, t_end, 1000)
    p_interp = np.interp(t_grid, p_ts, p_vals)
    h_interp = np.interp(t_grid, h_ts, h_vals)

    if len(t_grid) < 5:
        return {}

    r = float(np.corrcoef(p_interp, h_interp)[0, 1])
    return {
        "pupil_hr_correlation": r if np.isfinite(r) else 0.0,
    }


def cognitive_load_composite(rows: List[Dict], hr_data: List[Dict] = None,
                             baseline_pupil: float = None,
                             baseline_hr: float = None) -> Dict[str, float]:
    """
    Composite cognitive load index from pupil + fixation + blink + HR.
    Each component z-scored, then averaged.
    """
    components = {}

    # Pupil mean
    pupil_vals = []
    for r in rows:
        pl = _sf(r.get("pupil_left"))
        pr = _sf(r.get("pupil_right"))
        vals = [v for v in [pl, pr] if v is not None and v > 0]
        if vals:
            pupil_vals.append(float(np.mean(vals)))
    if pupil_vals:
        p_mean = float(np.mean(pupil_vals))
        if baseline_pupil and baseline_pupil > 0:
            components["pupil"] = (p_mean - baseline_pupil) / baseline_pupil
        else:
            components["pupil_raw"] = p_mean

    # Fixation duration mean
    fixations = _fixation_list(rows)
    if fixations:
        fix_durs = [f["duration"] for f in fixations]
        components["fix_dur"] = float(np.mean(fix_durs))

    # Blink rate (inverted — fewer blinks = more load)
    blink_f = blink_features(rows)
    if "blink_rate_per_min" in blink_f:
        components["inv_blink_rate"] = -blink_f["blink_rate_per_min"]

    # HR change from baseline
    if hr_data and baseline_hr:
        bpms = [_sf(r.get("bpm")) for r in hr_data]
        bpms = [b for b in bpms if b is not None]
        if bpms:
            components["hr_change"] = float(np.mean(bpms)) - baseline_hr

    feats = {}
    if components:
        # Z-score each component (can't truly z-score with one trial, but store raw)
        for k, v in components.items():
            feats[f"cogload_{k}"] = v
        feats["cogload_n_components"] = len(components)

    return feats


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def extract_gaze_features(rows: List[Dict], role: str, trial: str) -> Dict[str, Any]:
    """
    Extract all per-individual gaze features for one trial.

    Args:
        rows: gaze rows for this trial (from load_gaze_csv + filter_trial)
        role: 'director' or 'matcher'
        trial: trial label string
    """
    if not rows:
        return {"role": role, "trial": trial, "gaze_n_samples": 0}

    # Enrich SmartEye data if needed
    source = rows[0].get("source", "")
    if source == "smarteye":
        rows = _enrich_smarteye(rows)
    elif source == "aurora":
        rows = _detect_blinks_aurora(rows)

    result = {"role": role, "trial": trial, "gaze_n_samples": len(rows), "gaze_source": source}

    result.update(fixation_features(rows))
    result.update(saccade_features(rows))
    result.update(pupil_features(rows))
    result.update(blink_features(rows))
    result.update(scanpath_features(rows))
    result.update(aoi_features(rows))
    result.update(gaze_rqa_features(rows))
    result.update(data_quality_features(rows))

    return result


def extract_pair_gaze_features(rows_d: List[Dict], rows_m: List[Dict],
                                trial: str,
                                strokes: List[Dict] = None,
                                gt_json: Dict = None,
                                hr_d: List[Dict] = None,
                                hr_m: List[Dict] = None,
                                words_d: List[Dict] = None,
                                words_m: List[Dict] = None,
                                trial_start_ms: float = 0) -> Dict[str, Any]:
    """
    Extract all cross-participant and multimodal gaze features.

    Args:
        rows_d: Director gaze rows
        rows_m: Matcher gaze rows
        trial: trial label
        strokes: Matcher's stroke data [{points: [{x,y,t}]}]
        gt_json: Ground truth route JSON
        hr_d/hr_m: HR data for each role
        words_d/words_m: Whisper word timestamps for each role
        trial_start_ms: Unix ms of recording start
    """
    result = {"trial": trial}

    # Cross-participant gaze coordination
    result.update(gaze_convergence_features(rows_d, rows_m))
    result.update(gaze_crqa_features(rows_d, rows_m))
    result.update(gaze_coupling_lag(rows_d, rows_m))

    # Gaze-route alignment
    if gt_json:
        d_route = gaze_route_alignment(rows_d, gt_json, "director")
        result.update({f"dir_{k}": v for k, v in d_route.items()})
        m_route = gaze_route_alignment(rows_m, gt_json, "matcher")
        result.update({f"mat_{k}": v for k, v in m_route.items()})

    # Gaze-speech alignment (Director)
    if words_d:
        gs = gaze_speech_alignment(rows_d, words_d, "director", trial_start_ms)
        result.update({f"dir_{k}": v for k, v in gs.items()})

    # Gaze-drawing alignment (Matcher)
    if strokes:
        gd = gaze_drawing_alignment(rows_m, strokes, "matcher")
        result.update(gd)

    # Pupil-HR correlation
    if hr_d:
        phr_d = pupil_hr_correlation(rows_d, hr_d)
        result.update({f"dir_{k}": v for k, v in phr_d.items()})
    if hr_m:
        phr_m = pupil_hr_correlation(rows_m, hr_m)
        result.update({f"mat_{k}": v for k, v in phr_m.items()})

    # Cognitive load composite
    cl_d = cognitive_load_composite(rows_d, hr_d)
    result.update({f"dir_{k}": v for k, v in cl_d.items()})
    cl_m = cognitive_load_composite(rows_m, hr_m)
    result.update({f"mat_{k}": v for k, v in cl_m.items()})

    return result


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Extract gaze features from preprocessed CSV")
    ap.add_argument("--csv", required=True, help="Preprocessed eye CSV path")
    ap.add_argument("--role", required=True, choices=["director", "matcher"])
    ap.add_argument("--trial", default=None, help="Trial label (e.g., T03). If omitted, processes all.")
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    all_rows = load_gaze_csv(args.csv)

    if args.trial:
        trials = [args.trial]
    else:
        trials = sorted(set(r.get("trial", "") for r in all_rows if r.get("trial", "") != "no_trial"))

    results = []
    for t in trials:
        trial_rows = filter_trial(all_rows, t)
        if trial_rows:
            feats = extract_gaze_features(trial_rows, args.role, t)
            results.append(feats)
            print(f"Trial {t}: {len(trial_rows)} samples, {feats.get('fix_count', 0)} fixations")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote {len(results)} trial results to {args.out}")
    else:
        print(json.dumps(results[-1] if results else {}, indent=2, default=str))
