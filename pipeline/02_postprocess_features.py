#!/usr/bin/env python3
"""
Session ZIP → CSV dataset pipeline with GT scoring, HR features, prosody, and ASR (Smallest Pulse).

Usage:
  python scripts/postprocess.py \
    --zip /path/to/session.zip \
    --gt-dir "Ground Truth Maps" \
    --out out_dir \
    --smallest-key $SMALLEST_AI_KEY

Outputs in --out:
  metrics.csv, strokes.csv, hr_matcher.csv, hr_director.csv, hr_stats.csv (HRV+RQA),
  audio_manifest.csv, manifest.csv, prosody.csv, speech.csv (if ASR key).
"""

import argparse
import datetime
import io
import json
import math
import os
import shutil
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import librosa
import soundfile as sf
import requests

# Optional RQA
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

INF = 1e12

# Optional speech pipeline modules
try:
    from speech_features import extract_speech_features, extract_pair_features
    HAS_SPEECH = True
except Exception:
    HAS_SPEECH = False

try:
    from llm_eval import evaluate_trial as llm_evaluate_trial
    HAS_LLM_EVAL = True
except Exception:
    HAS_LLM_EVAL = False

try:
    from knowledge_graph import process_trial as kg_process_trial
    HAS_KG = True
except Exception:
    HAS_KG = False

try:
    from gaze_features import (load_gaze_csv, filter_trial as gaze_filter_trial,
                                extract_gaze_features, extract_pair_gaze_features)
    HAS_GAZE = True
except Exception:
    HAS_GAZE = False

try:
    from drawing_features import extract_drawing_features
    HAS_DRAWING = True
except Exception:
    HAS_DRAWING = False


def epoch_to_iso(t) -> str:
    """Convert Unix epoch milliseconds to ISO 8601 string, or empty if invalid."""
    if t is None or t == "":
        return ""
    try:
        return datetime.datetime.fromtimestamp(int(t) / 1000, tz=datetime.timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return ""


# ── Clock-offset extraction & timestamp alignment ──
# Convention: the merged ZIP contains clock_offset events from both roles.
#   - Director's event: offsetMs = Matcher_clock − Director_clock  (peerRole='matcher')
#   - Matcher's event:  offsetMs = Director_clock − Matcher_clock  (peerRole='director')
# We normalise all timestamps to **Director's clock** as reference.
# matcher_offset = Director_clock − Matcher_clock   (positive ⇒ Director ahead)
# To convert a Matcher timestamp:  t_ref = t_matcher + matcher_offset


def extract_clock_offset(zf, t_min: Optional[int] = None, t_max: Optional[int] = None) -> float:
    """Return matcher_offset (ms): Director_clock − Matcher_clock.

    Uses the MEDIAN of in-session clock_offset measurements (filtered by t_min/t_max
    when provided) to be robust against stale localStorage events. Drops zero-RTT
    timeouts and non-matcher peer measurements. Returns 0.0 if no offset found.
    """
    offsets: List[float] = []

    def _scan_events(events: list):
        for e in events:
            if e.get("type") != "clock_offset":
                continue
            et = e.get("t")
            # Filter by session window if provided
            if t_min is not None and et is not None and et < t_min:
                continue
            if t_max is not None and et is not None and et > t_max:
                continue
            p = e.get("payload", {})
            role = e.get("role", "")
            ms = p.get("offsetMs")
            if ms is None:
                continue
            ms = float(ms)
            # Drop zero-RTT timeouts (n=0 measurements)
            samples = p.get("samples", 0) or 0
            if samples == 0:
                continue
            peer = p.get("peerRole", "")
            # Normalise to Director_clock − Matcher_clock
            if role == "matcher" and peer == "director":
                offsets.append(ms)
            elif role == "director" and peer == "matcher":
                offsets.append(-ms)

    # 1) Session-level events (global events.json in merged ZIP)
    for path in ("session/events.json", "events.json"):
        try:
            raw = zf.read(path)
            _scan_events(json.loads(raw.decode("utf-8")))
        except (KeyError, Exception):
            pass

    # 2) Fall back: scan every trial's events.json
    if not offsets:
        for name in zf.namelist():
            if name.endswith("/events.json") and "trials/" in name:
                try:
                    raw = zf.read(name)
                    _scan_events(json.loads(raw.decode("utf-8")))
                except Exception:
                    pass

    if not offsets:
        return 0.0

    offsets_sorted = sorted(offsets)
    median = offsets_sorted[len(offsets_sorted) // 2]
    avg = sum(offsets) / len(offsets)
    spread = max(offsets) - min(offsets)
    print(f"[sync] Clock offset (Director − Matcher): median={median:.1f} ms, mean={avg:.1f} ms, spread={spread:.0f} ms ({len(offsets)} in-session measurement(s))")
    return median


def apply_offset_to_hr(rows: List[dict], offset_ms: float) -> List[dict]:
    """Shift all timestamps in HR rows by offset_ms.  Mutates in-place and returns."""
    if offset_ms == 0:
        return rows
    for r in rows:
        t = r.get("t")
        if isinstance(t, (int, float)):
            r["t"] = int(round(t + offset_ms))
    return rows


def interpolate_hr_pair(hr_m: List[dict], hr_d: List[dict],
                        sample_interval_ms: int = 1000) -> Tuple[List[float], List[float]]:
    """Resample two (already clock-aligned) HR series onto a common uniform
    time grid via linear interpolation.

    Returns (bpms_m, bpms_d) of equal length on the same 1-Hz grid spanning
    the overlapping time range of both series.
    """
    def _valid(rows):
        return [(r["t"], r["bpm"]) for r in rows
                if isinstance(r.get("t"), (int, float)) and isinstance(r.get("bpm"), (int, float, np.floating))]

    vm = _valid(hr_m)
    vd = _valid(hr_d)
    if len(vm) < 2 or len(vd) < 2:
        # Fall back to raw BPM lists (old behaviour)
        bm = [r["bpm"] for r in hr_m if isinstance(r.get("bpm"), (int, float, np.floating))]
        bd = [r["bpm"] for r in hr_d if isinstance(r.get("bpm"), (int, float, np.floating))]
        n = min(len(bm), len(bd))
        return bm[:n], bd[:n]

    vm.sort(key=lambda x: x[0])
    vd.sort(key=lambda x: x[0])

    # Overlapping window
    t_start = max(vm[0][0], vd[0][0])
    t_end = min(vm[-1][0], vd[-1][0])
    if t_end <= t_start:
        bm = [r["bpm"] for r in hr_m if isinstance(r.get("bpm"), (int, float, np.floating))]
        bd = [r["bpm"] for r in hr_d if isinstance(r.get("bpm"), (int, float, np.floating))]
        n = min(len(bm), len(bd))
        return bm[:n], bd[:n]

    grid = np.arange(t_start, t_end + 1, sample_interval_ms)
    if len(grid) < 2:
        bm = [r["bpm"] for r in hr_m if isinstance(r.get("bpm"), (int, float, np.floating))]
        bd = [r["bpm"] for r in hr_d if isinstance(r.get("bpm"), (int, float, np.floating))]
        n = min(len(bm), len(bd))
        return bm[:n], bd[:n]

    tm, bm = zip(*vm)
    td, bd = zip(*vd)
    interp_m = np.interp(grid, tm, bm).tolist()
    interp_d = np.interp(grid, td, bd).tolist()
    return interp_m, interp_d


def rescale_strokes(strokes: List[dict], target_w: int, target_h: int) -> List[dict]:
    max_x = max_y = 0
    for s in strokes:
        for p in s.get("points", []):
            if isinstance(p.get("x"), (int, float)):
                max_x = max(max_x, p["x"])
            if isinstance(p.get("y"), (int, float)):
                max_y = max(max_y, p["y"])
    src_w = 1024 if max_x > target_w else target_w
    src_h = 1024 if max_y > target_h else target_h
    if src_w == target_w and src_h == target_h:
        return strokes
    sx = target_w / src_w
    sy = target_h / src_h
    out = []
    for s in strokes:
        ns = dict(s)
        ns["points"] = [{**p, "x": p["x"] * sx, "y": p["y"] * sy} for p in s.get("points", [])]
        if "width" in ns and ns["width"]:
            ns["width"] = ns["width"] * min(sx, sy)
        out.append(ns)
    return out


def load_gt(gt_dir: str, map_number: int):
    path = os.path.join(gt_dir, f"gt_{map_number}.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def stroke_draw(draw: ImageDraw.ImageDraw, stroke: dict, erase: bool):
    pts = stroke.get("polyline") or stroke.get("points") or []
    if len(pts) < 2:
        return
    width = stroke.get("width", 3 if not erase else 20)
    xy = [(p["x"], p["y"]) for p in pts if isinstance(p, dict) and "x" in p and "y" in p]
    if len(xy) < 2:
        return
    color = 255 if erase else 0
    draw.line(xy, fill=color, width=int(width))


def strokes_to_mask(strokes: List[dict], width: int, height: int) -> np.ndarray:
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for s in strokes:
        mode = s.get("mode", "draw")
        erase = mode == "erase"
        stroke_draw(draw, s, erase)
    mask = np.array(img)
    return (mask < 250).astype(np.uint8)


def binary_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt_f = gt > 0
    pr_f = pred > 0
    tp = np.logical_and(gt_f, pr_f).sum()
    fp = np.logical_and(~gt_f, pr_f).sum()
    fn = np.logical_and(gt_f, ~pr_f).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "dice": f1}


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(np.float64) * 255.0
    pred = pred.astype(np.float64) * 255.0
    gt_f = gt.astype(np.float32)
    pr_f = pred.astype(np.float32)
    mu_x = gt_f.mean()
    mu_y = pr_f.mean()
    var_x = ((gt_f - mu_x) ** 2).mean()
    var_y = ((pr_f - mu_y) ** 2).mean()
    cov = ((gt_f - mu_x) * (pr_f - mu_y)).mean()
    C1 = 6.5025
    C2 = 58.5225
    return float(((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)))


def hausdorff(gt: np.ndarray, pred: np.ndarray) -> float:
    def coords(mask):
        yx = np.argwhere(mask > 0)
        return yx
    A = coords(gt)
    B = coords(pred)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    from scipy.spatial.distance import cdist
    D = cdist(A, B)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def chamfer(gt: np.ndarray, pred: np.ndarray) -> float:
    dt_gt = distance_transform_edt(gt == 0)
    dt_pr = distance_transform_edt(pred == 0)
    a = dt_gt[pred > 0]
    b = dt_pr[gt > 0]
    mean_a = float(a.mean()) if a.size else 0.0
    mean_b = float(b.mean()) if b.size else 0.0
    return float((mean_a + mean_b) / 2)


def boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue
            if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                out[y, x] = 1
                continue
            neigh = mask[y - 1:y + 2, x - 1:x + 2]
            if neigh.min() == 0:
                out[y, x] = 1
    return out


def boundary_f(gt: np.ndarray, pred: np.ndarray, tol: int = 2) -> Dict[str, float]:
    gt_b = boundary(gt)
    pr_b = boundary(pred)
    dt_gt = distance_transform_edt(gt_b == 0)
    dt_pr = distance_transform_edt(pr_b == 0)
    tot_p = pr_b.sum()
    tot_r = gt_b.sum()
    tp_p = (dt_gt[pr_b > 0] <= tol).sum() if tot_p else 0
    tp_r = (dt_pr[gt_b > 0] <= tol).sum() if tot_r else 0
    precision = tp_p / tot_p if tot_p else 0.0
    recall = tp_r / tot_r if tot_r else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_events(events: List[dict], map_number: int,
                 t_min: Optional[int] = None, t_max: Optional[int] = None) -> List[dict]:
    """Extract draw_stroke events. If t_min/t_max provided, filters by session timestamp window
    to exclude stale events leaked from previous sessions via localStorage backup."""
    out = []
    for idx, e in enumerate(events):
        if e.get("type") != "draw_stroke" or e.get("role") != "matcher":
            continue
        if e.get("payload", {}).get("mapNumber") not in (None, map_number):
            continue
        et = e.get("t")
        if t_min is not None and et is not None and et < t_min:
            continue
        if t_max is not None and et is not None and et > t_max:
            continue
        pts = e.get("payload", {}).get("polyline") or e.get("payload", {}).get("points") or []
        out.append({
            **e.get("payload", {}),
            "points": pts,
            "mode": e.get("payload", {}).get("mode", "draw"),
            "t": et,
            "strokeIndex": idx
        })
    return out


def derive_session_window(events: List[dict], hr_first_ts: Optional[int] = None,
                          margin_minutes: int = 60) -> tuple:
    """Derive (t_min, t_max) bounds for the current session.
    Uses HR first timestamp as primary anchor, falls back to trial_prepare events.
    Returns generous window: -1 hour to +3 hours from HR start."""
    if hr_first_ts is not None and hr_first_ts > 0:
        return (hr_first_ts - margin_minutes * 60_000,
                hr_first_ts + 3 * 3600_000)
    # Fallback: use most recent trial_prepare or trial_final_time event
    candidates = [e.get("t") for e in events
                  if e.get("type") in ("trial_prepare", "trial_final_time", "trial_success")
                  and e.get("t")]
    if candidates:
        last = max(candidates)
        return (last - 4 * 3600_000, last + 3600_000)
    return (None, None)


def csv_write(rows: List[dict], path: str):
    """Write rows to CSV, unioning headers across all rows."""
    if not rows:
        return
    headers = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = []
            for h in headers:
                v = r.get(h, "")
                if v is None:
                    v = ""
                s = str(v)
                if any(c in s for c in [",", "\"", "\n"]):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(",".join(vals) + "\n")


def convert_to_wav(audio_bytes: bytes, filename: str = "") -> bytes:
    import subprocess, tempfile
    ext = os.path.splitext(filename)[1] or ".webm"
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_out_path],
            capture_output=True, timeout=30, check=True,
        )
        with open(tmp_out_path, "rb") as f:
            wav_bytes = f.read()
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)
        return wav_bytes
    except Exception:
        try: os.unlink(tmp_in_path)
        except: pass
        try: os.unlink(tmp_out_path)
        except: pass
        return audio_bytes


def prosody_features(audio_bytes: bytes, sr_target: int = 16000) -> Dict[str, float]:
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr_target, mono=True)
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y).flatten()
        zcr = librosa.feature.zero_crossing_rate(y).flatten()
        f0 = librosa.yin(y, fmin=50, fmax=500)
        f0 = f0[np.isfinite(f0)]
        return {
            "duration_sec": duration,
            "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
            "rms_std": float(np.std(rms)) if rms.size else 0.0,
            "zcr_mean": float(np.mean(zcr)) if zcr.size else 0.0,
            "f0_mean": float(np.mean(f0)) if f0.size else 0.0,
            "f0_median": float(np.median(f0)) if f0.size else 0.0,
            "f0_coverage": float(f0.size / rms.size) if rms.size else 0.0,
        }
    except Exception:
        return {}


def asr_smallest(audio_bytes: bytes, api_key: str):
    url = "https://api.smallest.ai/waves/v1/pulse/get_text"
    params = {"language": "en", "word_timestamps": "true"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "audio/wav"}
    resp = requests.post(url, params=params, headers=headers, data=audio_bytes, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _safe_float(v, default: float = 0.0) -> float:
    """Convert a PyRQA result value to float, replacing NaN/Inf with *default*."""
    f = float(v)
    return f if np.isfinite(f) else default


def _ibi_from_bpm(bpms: List[float]) -> np.ndarray:
    """Convert BPM values to inter-beat intervals (IBI) in milliseconds."""
    return np.array([60000.0 / b for b in bpms if b > 0])


def _resample_uniform(values: np.ndarray, timestamps_ms: np.ndarray, fs: float = 4.0) -> np.ndarray:
    """Resample irregularly sampled data to uniform fs Hz via cubic interpolation."""
    if len(values) < 4 or len(timestamps_ms) < 4:
        return values
    from scipy.interpolate import CubicSpline
    t_sec = (timestamps_ms - timestamps_ms[0]) / 1000.0
    cs = CubicSpline(t_sec, values)
    t_uniform = np.arange(0, t_sec[-1], 1.0 / fs)
    return cs(t_uniform)


# ── Tier 1: Time-domain HRV ──

def _time_domain_hrv(bpms: List[float]) -> Dict[str, float]:
    """Compute time-domain HRV metrics from BPM series."""
    arr = np.array(bpms, dtype=np.float64)
    ibi = _ibi_from_bpm(bpms)
    diff_bpm = np.diff(arr)
    diff_ibi = np.diff(ibi)

    mean_hr = float(np.mean(arr))
    std_hr = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    mean_rr = float(np.mean(ibi)) if len(ibi) > 0 else 0.0
    sdnn = float(np.std(ibi, ddof=1)) if len(ibi) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_ibi ** 2))) if diff_ibi.size else 0.0
    ln_rmssd = float(np.log(rmssd)) if rmssd > 0 else 0.0
    nn50 = int(np.sum(np.abs(diff_ibi) > 50)) if diff_ibi.size else 0
    pnn50 = float(nn50 / len(diff_ibi)) if diff_ibi.size else 0.0
    sdsd = float(np.std(diff_ibi, ddof=1)) if len(diff_ibi) > 1 else 0.0
    hr_range = float(np.max(arr) - np.min(arr)) if len(arr) > 1 else 0.0

    return {
        "bpm_mean": mean_hr, "bpm_std": std_hr,
        "bpm_min": float(np.min(arr)), "bpm_max": float(np.max(arr)),
        "hr_range": hr_range,
        "mean_rr_ms": mean_rr, "sdnn_ms": sdnn,
        "rmssd_ms": rmssd, "ln_rmssd": ln_rmssd,
        "nn50": nn50, "pnn50": pnn50, "sdsd_ms": sdsd,
    }


# ── Tier 2: Nonlinear HRV ──

def _sample_entropy(data: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute Sample Entropy (SampEn). r = r_factor * std(data)."""
    N = len(data)
    if N < m + 2:
        return float('nan')
    r = r_factor * np.std(data, ddof=1)
    if r == 0:
        return float('nan')

    def _count_matches(template_len):
        count = 0
        templates = np.array([data[i:i + template_len] for i in range(N - template_len)])
        for i in range(len(templates)):
            # Chebyshev (max norm) distance
            dists = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(dists <= r)
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return float('nan')
    return -np.log(A / B) if A > 0 else float('nan')


def _dfa_alpha1(data: np.ndarray, min_n: int = 4, max_n: int = 16) -> float:
    """Detrended Fluctuation Analysis — short-term scaling exponent alpha1."""
    N = len(data)
    if N < max_n + 2:
        return float('nan')
    y = np.cumsum(data - np.mean(data))
    scales = np.arange(min_n, min(max_n + 1, N // 2))
    if len(scales) < 2:
        return float('nan')
    fluct = []
    for n in scales:
        n_segments = N // n
        if n_segments < 1:
            continue
        rms_list = []
        for seg in range(n_segments):
            idx = np.arange(seg * n, (seg + 1) * n)
            x = np.arange(n)
            coeffs = np.polyfit(x, y[idx], 1)
            trend = np.polyval(coeffs, x)
            rms_list.append(np.sqrt(np.mean((y[idx] - trend) ** 2)))
        if rms_list:
            fluct.append(np.mean(rms_list))
        else:
            fluct.append(0.0)
    fluct = np.array(fluct)
    valid = fluct > 0
    if np.sum(valid) < 2:
        return float('nan')
    log_n = np.log(scales[:len(fluct)][valid])
    log_f = np.log(fluct[valid])
    alpha = float(np.polyfit(log_n, log_f, 1)[0])
    return alpha


def _poincare(ibi: np.ndarray) -> Dict[str, float]:
    """Poincaré plot metrics: SD1, SD2, SD1/SD2 ratio."""
    if len(ibi) < 3:
        return {"sd1_ms": float('nan'), "sd2_ms": float('nan'), "sd1_sd2_ratio": float('nan')}
    diff = np.diff(ibi)
    sd1 = float(np.std(diff, ddof=1) / np.sqrt(2))
    sd2_sq = 2 * np.var(ibi, ddof=1) - 0.5 * np.var(diff, ddof=1)
    sd2 = float(np.sqrt(max(sd2_sq, 0)))
    ratio = float(sd1 / sd2) if sd2 > 0 else float('nan')
    return {"sd1_ms": sd1, "sd2_ms": sd2, "sd1_sd2_ratio": ratio}


def _nonlinear_hrv(bpms: List[float]) -> Dict[str, float]:
    """Compute nonlinear HRV metrics from BPM series."""
    ibi = _ibi_from_bpm(bpms)
    if len(ibi) < 10:
        return {}
    feats = {}
    feats["sample_entropy"] = _sample_entropy(ibi)
    feats["dfa_alpha1"] = _dfa_alpha1(ibi)
    feats.update(_poincare(ibi))
    return feats


# ── Tier 3: Frequency-domain HRV ──

def _freq_domain_hrv(bpms: List[float], timestamps_ms: np.ndarray = None) -> Dict[str, float]:
    """Compute frequency-domain HRV via Welch PSD on resampled IBI."""
    ibi = _ibi_from_bpm(bpms)
    if len(ibi) < 16:
        return {}
    fs = 4.0  # resample to 4 Hz
    if timestamps_ms is not None and len(timestamps_ms) == len(ibi):
        ibi_uniform = _resample_uniform(ibi, timestamps_ms, fs)
    else:
        timestamps_ms = np.cumsum(ibi)
        ibi_uniform = _resample_uniform(ibi, timestamps_ms, fs)
    if len(ibi_uniform) < 16:
        return {}
    from scipy.signal import welch
    nperseg = min(len(ibi_uniform), int(fs * 60))  # up to 60s window
    freqs, psd = welch(ibi_uniform - np.mean(ibi_uniform), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    vlf = float(np.trapz(psd[vlf_mask], dx=df)) if vlf_mask.any() else 0.0
    lf = float(np.trapz(psd[lf_mask], dx=df)) if lf_mask.any() else 0.0
    hf = float(np.trapz(psd[hf_mask], dx=df)) if hf_mask.any() else 0.0
    total = vlf + lf + hf
    lf_hf = float(lf / hf) if hf > 0 else float('nan')
    return {
        "vlf_power_ms2": vlf, "lf_power_ms2": lf, "hf_power_ms2": hf,
        "total_power_ms2": total, "lf_hf_ratio": lf_hf,
        "lf_nu": float(lf / (lf + hf) * 100) if (lf + hf) > 0 else float('nan'),
        "hf_nu": float(hf / (lf + hf) * 100) if (lf + hf) > 0 else float('nan'),
    }


# ── RQA (auto-recurrence, per individual) — full metric set ──

def _rqa_features(bpms: List[float], std: float) -> Dict[str, float]:
    """Full auto-RQA metric set using PyRQA."""
    if not HAS_RQA or len(bpms) < 10:
        return {}
    threshold = max(0.1 * std, 0.01) if std > 0 else 0.1
    try:
        ts = TimeSeries(bpms, embedding_dimension=2, time_delay=1)
        settings = Settings(ts, neighbourhood=FixedRadius(threshold),
                            similarity_measure=EuclideanMetric)
        comp = RQAComputation.create(settings)
        res = comp.run()
        return {
            "rqa_rr": _safe_float(res.recurrence_rate),
            "rqa_det": _safe_float(res.determinism),
            "rqa_mean_diag": _safe_float(res.average_diagonal_line),
            "rqa_max_diag": _safe_float(res.longest_diagonal_line),
            "rqa_div": _safe_float(1.0 / res.longest_diagonal_line if res.longest_diagonal_line > 0 else 0.0),
            "rqa_entr_diag": _safe_float(res.entropy_diagonal_lines),
            "rqa_lam": _safe_float(res.laminarity),
            "rqa_tt": _safe_float(res.trapping_time),
            "rqa_entr_vert": _safe_float(res.entropy_vertical_lines),
            "rqa_max_vert": _safe_float(res.longest_vertical_line),
        }
    except Exception:
        return {}


# ── Combined per-individual HR features ──

def hr_features(hr: List[dict], baseline_mean: float = None, baseline_n: int = 0) -> Dict[str, float]:
    """
    Comprehensive HRV: time-domain + nonlinear + frequency-domain + auto-RQA.
    If baseline_mean is provided, bpm values are mean-centered before RQA/nonlinear.
    """
    if not hr:
        return {}
    bpms = [r["bpm"] for r in hr if isinstance(r.get("bpm"), (int, float, np.floating))]
    if not bpms:
        return {}
    timestamps_ms = np.array([r["t"] for r in hr if isinstance(r.get("bpm"), (int, float, np.floating))
                              and isinstance(r.get("t"), (int, float))], dtype=np.float64)

    # Tier 1: Time-domain
    feats = _time_domain_hrv(bpms)
    feats["baseline_mean"] = baseline_mean if baseline_mean is not None else ""
    feats["baseline_n"] = baseline_n if baseline_mean is not None else ""

    # Optionally baseline-correct for RQA/nonlinear
    bpms_adj = [b - baseline_mean for b in bpms] if baseline_mean is not None else bpms
    std_adj = float(np.std(bpms_adj, ddof=1)) if len(bpms_adj) > 1 else 0.0

    # Tier 2: Nonlinear (baseline-adjusted for stationarity)
    feats.update(_nonlinear_hrv(bpms_adj))

    # Tier 3: Frequency-domain
    feats.update(_freq_domain_hrv(bpms, timestamps_ms if len(timestamps_ms) == len(bpms) else None))

    # Auto-RQA (full set)
    feats.update(_rqa_features(bpms_adj, std_adj))

    return feats


# ── CRQA (cross-recurrence) — full metric set ──

def crqa_features(bpms_m: List[float], bpms_d: List[float],
                  baseline_m: float = None, baseline_d: float = None) -> Dict[str, float]:
    """Cross-Recurrence Quantification Analysis between matcher and director HR — full metric set."""
    if not HAS_RQA or len(bpms_m) < 10 or len(bpms_d) < 10:
        return {}
    if baseline_m is not None:
        bpms_m = [b - baseline_m for b in bpms_m]
    if baseline_d is not None:
        bpms_d = [b - baseline_d for b in bpms_d]
    std_both = float(np.std(bpms_m + bpms_d, ddof=1))
    threshold = max(0.1 * std_both, 0.01) if std_both > 0 else 0.1
    try:
        ts_m = TimeSeries(bpms_m, embedding_dimension=2, time_delay=1)
        ts_d = TimeSeries(bpms_d, embedding_dimension=2, time_delay=1)
        settings = Settings((ts_m, ts_d),
                            analysis_type=Cross,
                            neighbourhood=FixedRadius(threshold),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=0)
        comp = RQAComputation.create(settings)
        res = comp.run()
        return {
            "crqa_rr": _safe_float(res.recurrence_rate),
            "crqa_det": _safe_float(res.determinism),
            "crqa_mean_diag": _safe_float(res.average_diagonal_line),
            "crqa_max_diag": _safe_float(res.longest_diagonal_line),
            "crqa_div": _safe_float(1.0 / res.longest_diagonal_line if res.longest_diagonal_line > 0 else 0.0),
            "crqa_entr_diag": _safe_float(res.entropy_diagonal_lines),
            "crqa_lam": _safe_float(res.laminarity),
            "crqa_tt": _safe_float(res.trapping_time),
            "crqa_entr_vert": _safe_float(res.entropy_vertical_lines),
            "crqa_max_vert": _safe_float(res.longest_vertical_line),
        }
    except Exception:
        return {}


# ── MdRQA (Multidimensional RQA) — joint phase space ──

def mdrqa_features(bpms_m: List[float], bpms_d: List[float],
                   baseline_m: float = None, baseline_d: float = None) -> Dict[str, float]:
    """MdRQA: embed both HR series in joint 2D phase space."""
    if not HAS_RQA or len(bpms_m) < 10 or len(bpms_d) < 10:
        return {}
    if baseline_m is not None:
        bpms_m = [b - baseline_m for b in bpms_m]
    if baseline_d is not None:
        bpms_d = [b - baseline_d for b in bpms_d]
    # Align lengths
    n = min(len(bpms_m), len(bpms_d))
    bpms_m, bpms_d = bpms_m[:n], bpms_d[:n]
    # Z-score normalize each channel independently
    arr_m = np.array(bpms_m, dtype=np.float64)
    arr_d = np.array(bpms_d, dtype=np.float64)
    std_m = np.std(arr_m, ddof=1)
    std_d = np.std(arr_d, ddof=1)
    if std_m > 0:
        arr_m = (arr_m - np.mean(arr_m)) / std_m
    if std_d > 0:
        arr_d = (arr_d - np.mean(arr_d)) / std_d
    # Build joint distance matrix (Euclidean in 2D)
    joint = np.column_stack([arr_m, arr_d])  # shape (n, 2)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(joint, metric='euclidean'))
    # Threshold: target ~5% recurrence rate
    threshold = np.percentile(D[np.triu_indices_from(D, k=1)], 5)
    if threshold <= 0:
        threshold = 0.01
    RP = (D <= threshold).astype(np.uint8)
    np.fill_diagonal(RP, 0)  # exclude line of identity
    return _rp_metrics(RP, prefix="mdrqa_")


def _rp_metrics(RP: np.ndarray, prefix: str = "", min_line: int = 2) -> Dict[str, float]:
    """Extract RQA metrics directly from a recurrence plot matrix."""
    N = RP.shape[0]
    total_pts = N * (N - 1)  # excluding diagonal
    rec_pts = int(RP.sum())
    rr = rec_pts / total_pts if total_pts > 0 else 0.0

    # Diagonal lines
    diag_lengths = []
    for k in range(-N + 1, N):
        if k == 0:
            continue
        diag = np.diag(RP, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= min_line:
                    diag_lengths.append(length)
                length = 0
        if length >= min_line:
            diag_lengths.append(length)

    # Vertical lines
    vert_lengths = []
    for col in range(N):
        length = 0
        for row in range(N):
            if RP[row, col]:
                length += 1
            else:
                if length >= min_line:
                    vert_lengths.append(length)
                length = 0
        if length >= min_line:
            vert_lengths.append(length)

    det_pts = sum(diag_lengths) if diag_lengths else 0
    det = det_pts / rec_pts if rec_pts > 0 else 0.0
    mean_diag = float(np.mean(diag_lengths)) if diag_lengths else 0.0
    max_diag = float(max(diag_lengths)) if diag_lengths else 0.0
    div = 1.0 / max_diag if max_diag > 0 else 0.0

    # Shannon entropy of diagonal line lengths
    if diag_lengths:
        hist = np.bincount(diag_lengths)
        hist = hist[hist > 0].astype(float)
        probs = hist / hist.sum()
        entr_diag = float(-np.sum(probs * np.log(probs)))
    else:
        entr_diag = 0.0

    lam_pts = sum(vert_lengths) if vert_lengths else 0
    lam = lam_pts / rec_pts if rec_pts > 0 else 0.0
    tt = float(np.mean(vert_lengths)) if vert_lengths else 0.0
    max_vert = float(max(vert_lengths)) if vert_lengths else 0.0

    if vert_lengths:
        hist_v = np.bincount(vert_lengths)
        hist_v = hist_v[hist_v > 0].astype(float)
        probs_v = hist_v / hist_v.sum()
        entr_vert = float(-np.sum(probs_v * np.log(probs_v)))
    else:
        entr_vert = 0.0

    return {
        f"{prefix}rr": rr, f"{prefix}det": det,
        f"{prefix}mean_diag": mean_diag, f"{prefix}max_diag": max_diag,
        f"{prefix}div": div, f"{prefix}entr_diag": entr_diag,
        f"{prefix}lam": lam, f"{prefix}tt": tt,
        f"{prefix}max_vert": max_vert, f"{prefix}entr_vert": entr_vert,
    }


# ── DCRP (Diagonal Cross-Recurrence Profile) — leader-follower ──

def dcrp_features(bpms_m: List[float], bpms_d: List[float],
                  baseline_m: float = None, baseline_d: float = None,
                  max_lag: int = 20) -> Dict[str, float]:
    """Diagonal Cross-Recurrence Profile: %REC along diagonals offset from LoS."""
    n = min(len(bpms_m), len(bpms_d))
    if n < 10:
        return {}
    arr_m = np.array(bpms_m[:n], dtype=np.float64)
    arr_d = np.array(bpms_d[:n], dtype=np.float64)
    if baseline_m is not None:
        arr_m = arr_m - baseline_m
    if baseline_d is not None:
        arr_d = arr_d - baseline_d
    std_both = np.std(np.concatenate([arr_m, arr_d]), ddof=1)
    threshold = max(0.1 * std_both, 0.01) if std_both > 0 else 0.1

    # Build cross-distance matrix
    from scipy.spatial.distance import cdist
    D = cdist(arr_m.reshape(-1, 1), arr_d.reshape(-1, 1), metric='euclidean')
    CRP = (D <= threshold).astype(np.uint8)

    profile = {}
    for lag in range(-max_lag, max_lag + 1):
        diag = np.diag(CRP, lag)
        profile[lag] = float(diag.mean()) if len(diag) > 0 else 0.0

    # Find peak
    peak_lag = max(profile, key=profile.get)
    peak_rr = profile[peak_lag]
    # Profile width at half-max
    half_max = peak_rr / 2
    above = [lag for lag, v in profile.items() if v >= half_max]
    width = (max(above) - min(above)) if above else 0

    return {
        "dcrp_peak_lag": float(peak_lag),
        "dcrp_peak_rr": peak_rr,
        "dcrp_width": float(width),
        "dcrp_los_rr": profile.get(0, 0.0),  # %REC at lag 0 (line of synchrony)
    }


# ── Windowed Cross-Correlation ──

def windowed_xcorr(bpms_m: List[float], bpms_d: List[float],
                   window_sec: float = 30.0, step_sec: float = 10.0,
                   sample_interval_sec: float = 1.0) -> Dict[str, float]:
    """Sliding-window Pearson cross-correlation with peak lag detection."""
    n = min(len(bpms_m), len(bpms_d))
    if n < 10:
        return {}
    arr_m = np.array(bpms_m[:n], dtype=np.float64)
    arr_d = np.array(bpms_d[:n], dtype=np.float64)
    win_samples = max(int(window_sec / sample_interval_sec), 5)
    step_samples = max(int(step_sec / sample_interval_sec), 1)

    correlations = []
    for start in range(0, n - win_samples + 1, step_samples):
        seg_m = arr_m[start:start + win_samples]
        seg_d = arr_d[start:start + win_samples]
        if np.std(seg_m) < 1e-6 or np.std(seg_d) < 1e-6:
            continue
        r = float(np.corrcoef(seg_m, seg_d)[0, 1])
        if np.isfinite(r):
            correlations.append(r)

    if not correlations:
        return {}
    return {
        "wcc_mean_r": float(np.mean(correlations)),
        "wcc_max_r": float(np.max(correlations)),
        "wcc_min_r": float(np.min(correlations)),
        "wcc_std_r": float(np.std(correlations)),
        "wcc_n_windows": len(correlations),
        "wcc_pct_positive": float(sum(1 for r in correlations if r > 0) / len(correlations)) if correlations else 0.0,
    }


# ── Transfer Entropy (symbolic) ──

def _symbolic_transfer_entropy(x: np.ndarray, y: np.ndarray, m: int = 3) -> float:
    """Symbolic Transfer Entropy: X → Y (does past of X help predict future of Y?)."""
    N = len(x)
    if N < m + 2:
        return float('nan')
    # Symbolize: 0 = decrease, 1 = increase
    sx = (np.diff(x) > 0).astype(int)
    sy = (np.diff(y) > 0).astype(int)
    n = min(len(sx), len(sy))
    sx, sy = sx[:n], sy[:n]
    if n < m + 1:
        return float('nan')

    from collections import Counter
    # Count joint patterns: (y_future, y_past_m, x_past_m)
    joint_yyx = Counter()
    joint_yy = Counter()
    joint_yx = Counter()
    marg_y = Counter()
    for i in range(m, n):
        y_fut = sy[i]
        y_past = tuple(sy[i - m:i])
        x_past = tuple(sx[i - m:i])
        joint_yyx[(y_fut, y_past, x_past)] += 1
        joint_yy[(y_fut, y_past)] += 1
        joint_yx[(y_past, x_past)] += 1
        marg_y[y_past] += 1

    total = sum(joint_yyx.values())
    if total == 0:
        return float('nan')
    te = 0.0
    for (yf, yp, xp), count in joint_yyx.items():
        p_yyx = count / total
        p_yy = joint_yy[(yf, yp)] / total
        p_yx = joint_yx[(yp, xp)] / total
        p_y = marg_y[yp] / total
        if p_yy > 0 and p_yx > 0 and p_y > 0:
            ratio = (p_yyx * p_y) / (p_yy * p_yx)
            if ratio > 0:
                te += p_yyx * np.log2(ratio)
    return float(te)


def transfer_entropy_features(bpms_m: List[float], bpms_d: List[float]) -> Dict[str, float]:
    """Bidirectional symbolic transfer entropy."""
    arr_m = np.array(bpms_m, dtype=np.float64)
    arr_d = np.array(bpms_d, dtype=np.float64)
    n = min(len(arr_m), len(arr_d))
    if n < 10:
        return {}
    arr_m, arr_d = arr_m[:n], arr_d[:n]
    te_m2d = _symbolic_transfer_entropy(arr_m, arr_d)  # matcher → director
    te_d2m = _symbolic_transfer_entropy(arr_d, arr_m)  # director → matcher
    asym = te_d2m - te_m2d if np.isfinite(te_d2m) and np.isfinite(te_m2d) else float('nan')
    return {
        "te_matcher_to_director": te_m2d,
        "te_director_to_matcher": te_d2m,
        "te_asymmetry": asym,  # positive = director leads
    }


# ── Windowed CRQA (Tier 3) ──

def windowed_crqa(bpms_m: List[float], bpms_d: List[float],
                  window_sec: float = 60.0, step_sec: float = 30.0,
                  sample_interval_sec: float = 1.0) -> Dict[str, float]:
    """Sliding-window CRQA: track synchrony evolution within a trial."""
    n = min(len(bpms_m), len(bpms_d))
    win_samples = max(int(window_sec / sample_interval_sec), 10)
    step_samples = max(int(step_sec / sample_interval_sec), 1)
    if n < win_samples:
        return {}

    rr_values, det_values = [], []
    for start in range(0, n - win_samples + 1, step_samples):
        seg_m = bpms_m[start:start + win_samples]
        seg_d = bpms_d[start:start + win_samples]
        feats = crqa_features(seg_m, seg_d)
        if feats:
            rr_values.append(feats.get("crqa_rr", 0.0))
            det_values.append(feats.get("crqa_det", 0.0))

    if not rr_values:
        return {}
    return {
        "wcrqa_rr_mean": float(np.mean(rr_values)),
        "wcrqa_rr_std": float(np.std(rr_values)),
        "wcrqa_rr_trend": float(np.polyfit(range(len(rr_values)), rr_values, 1)[0]) if len(rr_values) > 1 else 0.0,
        "wcrqa_det_mean": float(np.mean(det_values)),
        "wcrqa_det_std": float(np.std(det_values)),
        "wcrqa_det_trend": float(np.polyfit(range(len(det_values)), det_values, 1)[0]) if len(det_values) > 1 else 0.0,
        "wcrqa_n_windows": len(rr_values),
    }


# ── Surrogate baseline (pseudo-dyad) ──

def surrogate_crqa(bpms_m: List[float], bpms_d: List[float],
                   n_surrogates: int = 20,
                   baseline_m: float = None, baseline_d: float = None) -> Dict[str, float]:
    """Generate surrogate (time-shifted) baselines for CRQA significance testing."""
    if len(bpms_m) < 10 or len(bpms_d) < 10:
        return {}
    real = crqa_features(bpms_m, bpms_d, baseline_m, baseline_d)
    if not real:
        return {}
    rng = np.random.default_rng(42)
    surr_rrs, surr_dets = [], []
    d_arr = np.array(bpms_d)
    for _ in range(n_surrogates):
        shift = rng.integers(len(d_arr) // 4, 3 * len(d_arr) // 4)
        d_shifted = list(np.roll(d_arr, shift))
        s = crqa_features(bpms_m, d_shifted, baseline_m, baseline_d)
        if s:
            surr_rrs.append(s.get("crqa_rr", 0.0))
            surr_dets.append(s.get("crqa_det", 0.0))
    if not surr_rrs:
        return {}
    real_rr = real.get("crqa_rr", 0.0)
    real_det = real.get("crqa_det", 0.0)
    return {
        "surr_rr_mean": float(np.mean(surr_rrs)),
        "surr_rr_std": float(np.std(surr_rrs)),
        "surr_rr_z": float((real_rr - np.mean(surr_rrs)) / np.std(surr_rrs)) if np.std(surr_rrs) > 0 else 0.0,
        "surr_det_mean": float(np.mean(surr_dets)),
        "surr_det_std": float(np.std(surr_dets)),
        "surr_det_z": float((real_det - np.mean(surr_dets)) / np.std(surr_dets)) if np.std(surr_dets) > 0 else 0.0,
        "surr_n": len(surr_rrs),
    }


def parse_hr_csv(csv_text: str, role: str, trial: int) -> List[dict]:
    if not csv_text:
        return []
    lines = csv_text.strip().splitlines()
    if len(lines) <= 1:
        return []
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 3:
            continue
        t = int(parts[0]) if parts[0].isdigit() else None
        try:
            bpm = float(parts[2]) if parts[2] else None
        except (ValueError, TypeError):
            bpm = None
        phase = parts[3] if len(parts) > 3 else ""
        rows.append({"t": t, "bpm": bpm, "phase": phase, "role": role, "trial": trial})
    return rows


def build_ml_table(metrics_rows, hr_stats_rows, drawing_feat_rows,
                   gaze_feat_rows, gaze_pair_rows, speech_feat_rows,
                   pair_speech_rows, llm_eval_rows, manifest_rows,
                   prosody_rows=None, kg_rows=None) -> List[Dict]:
    """Join all per-trial feature tables into one ML-ready row per (sessionId, trial).

    Each feature group is prefixed so columns never collide:
      (no prefix)          — manifest metadata, mapNumber, TLX, timing
      map_                 — map accuracy metrics (iou, f1, ssim, hausdorff, …)
      hr_matcher_ /
      hr_director_ /
      hr_cross_            — per-role HRV + auto/cross RQA + CRQA/MdRQA/TE
      draw_                — drawing behaviour features
      gaze_director_ /
      gaze_matcher_        — per-role gaze features
      gaze_pair_           — cross-participant gaze features
      prosody_director_ /
      prosody_matcher_     — librosa prosody (duration, RMS, ZCR, F0)
      speech_director_ /
      speech_matcher_      — Whisper + Parselmouth + OpenSMILE speech features
      speech_pair_         — dyadic speech features
      llm_                 — LLM dialogue evaluation scores
      kg_                  — spatial knowledge graph features
    """
    from collections import defaultdict as _dd
    table: Dict[tuple, dict] = _dd(dict)
    # Fields that are indexing / file metadata — never useful as ML features
    SKIP = {"sessionId", "trial", "role", "filename", "bytes", "error",
            "ref_clock", "t_iso", "trial_start_iso", "trial_end_iso"}

    def _norm_trial(t):
        """Normalize trial label: 'T01' → 1, '1' → 1, 1 → 1."""
        if t is None or t == "":
            return ""
        if isinstance(t, int):
            return t
        s = str(t).strip()
        if s.startswith("T") or s.startswith("t"):
            s = s[1:]
        try:
            return int(s)
        except (ValueError, TypeError):
            return s  # fallback: keep as string

    def _key(r):
        return (r.get("sessionId", ""), _norm_trial(r.get("trial", "")))

    def _base(r):
        k = _key(r)
        table[k]["sessionId"] = r.get("sessionId", "")
        table[k]["trial"] = _norm_trial(r.get("trial", ""))

    def _add(r, pfx):
        k = _key(r)
        for field, val in r.items():
            if field in SKIP:
                continue
            col = pfx + field
            # Don't overwrite an already-set non-empty value with empty
            if col not in table[k] or table[k][col] == "" or table[k][col] is None:
                table[k][col] = val

    # Manifest: base metadata + TLX (no prefix — these are covariates, not features)
    for r in manifest_rows:
        _base(r); _add(r, "")

    # Map accuracy
    for r in metrics_rows:
        _base(r); _add(r, "map_")

    # HR / HRV / RQA — pivot by role (role values: matcher, director, cross)
    for r in hr_stats_rows:
        _base(r)
        role = r.get("role", "")
        _add(r, f"hr_{role}_" if role else "hr_")

    # Drawing behaviour
    for r in drawing_feat_rows:
        _base(r); _add(r, "draw_")

    # Per-role gaze
    for r in gaze_feat_rows:
        _base(r)
        role = r.get("role", "")
        _add(r, f"gaze_{role}_" if role else "gaze_")

    # Pair gaze
    for r in gaze_pair_rows:
        _base(r); _add(r, "gaze_pair_")

    # Librosa prosody — determine role from audio filename
    for r in (prosody_rows or []):
        fn = r.get("filename", "").lower()
        role = "director" if "director" in fn else ("matcher" if "matcher" in fn else None)
        if role is None:
            continue
        _base(r)
        _add(r, f"prosody_{role}_")

    # Per-role speech features (Whisper + Parselmouth + OpenSMILE)
    for r in speech_feat_rows:
        _base(r)
        role = r.get("role", "")
        _add(r, f"speech_{role}_" if role else "speech_")

    # Pair speech
    for r in pair_speech_rows:
        _base(r); _add(r, "speech_pair_")

    # LLM evaluation
    for r in llm_eval_rows:
        _base(r); _add(r, "llm_")

    # Spatial knowledge graph
    for r in (kg_rows or []):
        _base(r); _add(r, "kg_")

    return list(table.values())


def process_zip(zip_path: str, gt_dir: str, out_dir: str, asr_key: str = None,
                openai_key: str = None, map_image_dir: str = None,
                eye_csv_director: str = None, eye_csv_matcher: str = None):
    os.makedirs(out_dir, exist_ok=True)

    # Pre-load eye tracking data if available
    gaze_data_d = load_gaze_csv(eye_csv_director) if HAS_GAZE and eye_csv_director else None
    gaze_data_m = load_gaze_csv(eye_csv_matcher) if HAS_GAZE and eye_csv_matcher else None

    # Copy eye CSVs into output dir so they're included in the output ZIP
    if eye_csv_director and os.path.exists(eye_csv_director):
        shutil.copy(eye_csv_director, os.path.join(out_dir, "eye_director.csv"))
    if eye_csv_matcher and os.path.exists(eye_csv_matcher):
        shutil.copy(eye_csv_matcher, os.path.join(out_dir, "eye_matcher.csv"))

    with zipfile.ZipFile(zip_path, "r") as zf:
        trial_dirs = sorted({p.split("/")[1] for p in zf.namelist() if p.startswith("trials/") and len(p.split("/")) > 2})
        metrics_rows = []
        strokes_rows = []
        hr_matcher_rows = []
        hr_director_rows = []
        hr_stats_rows = []
        audio_rows = []
        manifest_rows = []
        prosody_rows = []
        speech_rows = []
        speech_feat_rows = []   # Whisper + Parselmouth + OpenSMILE
        pair_speech_rows = []   # Dyadic speech features
        llm_eval_rows = []      # LLM dialogue evaluation
        kg_rows = []            # Knowledge graph features
        gaze_feat_rows = []     # Per-individual gaze features
        gaze_pair_rows = []     # Cross-participant gaze features
        drawing_feat_rows = []  # Drawing behavior features
        gt_cache = {}
        ts_rows = []
        audio_out_dir = os.path.join(out_dir, "audio")
        os.makedirs(audio_out_dir, exist_ok=True)

        zip_baseline = {"director": None, "matcher": None}
        try:
            bl_bytes = zf.read("session/hr_baseline.json")
            bl = json.loads(bl_bytes.decode("utf-8"))
            if isinstance(bl.get("director"), (int, float)):
                zip_baseline["director"] = float(bl["director"])
            if isinstance(bl.get("matcher"), (int, float)):
                zip_baseline["matcher"] = float(bl["matcher"])
        except Exception:
            pass

        # ── Determine session window from HR (anchors stale-event filtering) ──
        session_t_min = session_t_max = None
        try:
            for td in trial_dirs[:3]:  # try first few trials
                hb = zf.read(f"trials/{td}/hr/hr_director.csv").decode().strip().split("\n")
                if len(hb) > 1:
                    try:
                        first_ts = int(hb[1].split(",")[0])
                        session_t_min = first_ts - 60 * 60_000  # 1 hour before HR start
                        session_t_max = first_ts + 3 * 3600_000  # 3 hours after
                        print(f"[session] Window: {session_t_min} → {session_t_max} (anchored to HR start {first_ts})")
                        break
                    except (ValueError, IndexError):
                        continue
        except Exception:
            pass

        # ── Clock offset: align Matcher timestamps to Director's reference clock ──
        matcher_offset_ms = extract_clock_offset(zf, t_min=session_t_min, t_max=session_t_max)

        # Apply clock offset to Matcher gaze data (preprocess_eye.py aligned it
        # to Matcher's laptop clock; we now shift to Director's reference clock)
        if gaze_data_m and matcher_offset_ms != 0:
            for gr in gaze_data_m:
                t = gr.get("t_unix_ms")
                if t is not None and t != "":
                    try:
                        corrected = int(round(int(t) + matcher_offset_ms))
                        gr["t_unix_ms"] = str(corrected)
                        gr["t_iso"] = epoch_to_iso(corrected)
                    except (ValueError, TypeError):
                        pass
            print(f"[sync] Applied {matcher_offset_ms:+.1f}ms offset to {len(gaze_data_m)} Matcher gaze samples")

        for tdir in trial_dirs:
            try:
                trial_idx = int(tdir.replace("T", ""))
            except (ValueError, TypeError):
                continue
            def read(path):
                try:
                    with zf.open(path) as f:
                        return f.read()
                except KeyError:
                    return None

            events_bytes = read(f"trials/{tdir}/events.json")
            strokes_bytes = read(f"trials/{tdir}/strokes.json")
            hr_m_bytes = read(f"trials/{tdir}/hr/hr_matcher.csv")
            hr_d_bytes = read(f"trials/{tdir}/hr/hr_director.csv")

            events = json.loads(events_bytes.decode("utf-8")) if events_bytes else []

            # Map number: prefer in-session trial_prepare/trial_success/trial_final_time events
            # over arbitrary events (which may be stale from previous localStorage sessions)
            map_number = None
            authoritative_types = ("trial_prepare", "trial_success", "trial_final_time")
            for e in reversed(events):
                if e.get("type") not in authoritative_types:
                    continue
                et = e.get("t")
                if session_t_min is not None and et is not None and et < session_t_min:
                    continue
                if session_t_max is not None and et is not None and et > session_t_max:
                    continue
                if e.get("payload", {}).get("trialIndex") != trial_idx:
                    continue
                mn = e.get("payload", {}).get("mapNumber")
                if isinstance(mn, int):
                    map_number = mn
                    break
            # Fallback: any in-session event with this trial's mapNumber
            if map_number is None:
                for e in reversed(events):
                    et = e.get("t")
                    if session_t_min is not None and et is not None and et < session_t_min:
                        continue
                    if session_t_max is not None and et is not None and et > session_t_max:
                        continue
                    mn = e.get("payload", {}).get("mapNumber")
                    if isinstance(mn, int):
                        map_number = mn
                        break
            # Last resort: any event (legacy behavior)
            if map_number is None:
                for e in reversed(events):
                    mn = e.get("payload", {}).get("mapNumber")
                    if isinstance(mn, int):
                        map_number = mn
                        break
            if map_number is None:
                continue

            # Parse strokes filtered to current session (excludes stale localStorage events)
            raw_strokes = parse_events(events, map_number, t_min=session_t_min, t_max=session_t_max)
            if not raw_strokes and strokes_bytes:
                try:
                    st = json.loads(strokes_bytes.decode("utf-8"))
                    if isinstance(st, list):
                        raw_strokes = st
                except Exception:
                    pass
            # filter drawable
            strokes = []
            for s in raw_strokes:
                pts = s.get("points") or s.get("polyline") or []
                pts = [p for p in pts if isinstance(p, dict) and "x" in p and "y" in p]
                if len(pts) < 2:
                    continue
                mode = s.get("mode", "draw")
                if mode not in ("draw", "erase"):
                    continue
                strokes.append({**s, "points": pts, "mode": mode})

            if map_number not in gt_cache:
                gt_cache[map_number] = load_gt(gt_dir, map_number)
            gt = gt_cache[map_number]
            width = gt.get("image", {}).get("width", 651)
            height = gt.get("image", {}).get("height", 900)
            strokes = rescale_strokes(strokes, width, height)
            gt_mask = strokes_to_mask(gt.get("strokes", []), width, height)
            pred_mask = strokes_to_mask(strokes, width, height)

            m = binary_metrics(gt_mask, pred_mask)
            s = ssim(gt_mask, pred_mask)
            h = hausdorff(gt_mask, pred_mask)
            cd = chamfer(gt_mask, pred_mask)
            bf = boundary_f(gt_mask, pred_mask, 2)
            coverage_gt = float(gt_mask.mean())
            coverage_pred = float(pred_mask.mean())

            metrics_rows.append({
                "sessionId": os.path.basename(zip_path),
                "trial": trial_idx,
                "mapNumber": map_number,
                **m,
                "ssim": s,
                "hausdorff": h,
                "chamfer": cd,
                "boundary_f1": bf["f1"],
                "boundary_p": bf["precision"],
                "boundary_r": bf["recall"],
                "coverage_gt": coverage_gt,
                "coverage_pred": coverage_pred,
            })

            # Strokes are Matcher-sourced — shift timestamps to Director reference clock
            # This must happen BEFORE drawing features so temporal features are on a
            # consistent clock with HR and gaze data.
            def _offset_t(t_val):
                """Apply matcher_offset_ms to a timestamp value."""
                if matcher_offset_ms == 0 or t_val == "" or t_val is None:
                    return t_val
                try:
                    return int(round(int(t_val) + matcher_offset_ms))
                except (ValueError, TypeError):
                    return t_val

            if matcher_offset_ms != 0:
                for sraw in strokes:
                    sraw["t"] = _offset_t(sraw.get("t", ""))
                    for p in sraw.get("points", []):
                        if isinstance(p, dict) and "t" in p:
                            p["t"] = _offset_t(p["t"])

            # Drawing behavior features (strokes now on Director's reference clock)
            if HAS_DRAWING and strokes:
                df_row = extract_drawing_features(strokes, trial_idx, gt_cache.get(map_number))
                df_row["sessionId"] = os.path.basename(zip_path)
                df_row["mapNumber"] = map_number
                drawing_feat_rows.append(df_row)

            for sidx, sraw in enumerate(strokes):
                stroke_t = sraw.get("t", "")
                for pidx, p in enumerate(sraw.get("points", [])):
                    pt_t = p.get("t", "") if isinstance(p, dict) else ""
                    raw_t = pt_t if pt_t else sraw.get("t", "")
                    strokes_rows.append({
                        "sessionId": os.path.basename(zip_path),
                        "trial": trial_idx,
                        "mapNumber": map_number,
                        "strokeIndex": sraw.get("strokeIndex", sidx),
                        "pointIndex": pidx,
                        "t_unix_ms": raw_t,
                        "t_iso": epoch_to_iso(raw_t),
                        "stroke_t_unix_ms": stroke_t,
                        "mode": sraw.get("mode", "draw"),
                        "x": p["x"],
                        "y": p["y"],
                    })

            # HR — parse and align Matcher timestamps to Director's reference clock
            hr_m = parse_hr_csv(hr_m_bytes.decode("utf-8"), "matcher", trial_idx) if hr_m_bytes else []
            hr_d = parse_hr_csv(hr_d_bytes.decode("utf-8"), "director", trial_idx) if hr_d_bytes else []
            apply_offset_to_hr(hr_m, matcher_offset_ms)

            def add_hr_rows(role_rows, rows, role):
                baseline_mean = zip_baseline.get(role)
                baseline_n = 1 if baseline_mean is not None else 0
                for r in rows:
                    t_val = r.get("t", "")
                    role_rows.append({
                        "sessionId": os.path.basename(zip_path),
                        "trial": trial_idx,
                        "role": role,
                        "kind": "raw",
                        "t_unix_ms": t_val,
                        "t_iso": epoch_to_iso(t_val),
                        "bpm": r.get("bpm", ""),
                        "phase": r.get("phase", ""),
                        "n": "",
                        "bpm_mean": "",
                        "bpm_min": "",
                        "bpm_max": "",
                        "bpm_std": "",
                    })
                bpms = [r["bpm"] for r in rows if isinstance(r.get("bpm"), (int, float, np.floating))]
                if bpms:
                    role_rows.append({
                        "sessionId": os.path.basename(zip_path),
                        "trial": trial_idx,
                        "role": role,
                        "kind": "summary",
                        "t_unix_ms": "",
                        "t_iso": "",
                        "bpm": "",
                        "phase": "",
                        "n": len(bpms),
                        "bpm_mean": float(np.mean(bpms)),
                        "bpm_min": float(np.min(bpms)),
                        "bpm_max": float(np.max(bpms)),
                        "bpm_std": float(np.std(bpms)),
                        "baseline_mean": baseline_mean if baseline_mean is not None else "",
                        "baseline_n": baseline_n if baseline_mean is not None else "",
                    })
                    hf = hr_features(rows, baseline_mean=baseline_mean, baseline_n=baseline_n)
                    if hf:
                        stats_row = {"sessionId": os.path.basename(zip_path), "trial": trial_idx, "role": role}
                        stats_row.update(hf)
                        hr_stats_rows.append(stats_row)

            add_hr_rows(hr_matcher_rows, hr_m, "matcher")
            add_hr_rows(hr_director_rows, hr_d, "director")

            # Cross-dyad metrics — timestamps already aligned to Director clock;
            # interpolate both series to a common 1 Hz grid for proper temporal alignment
            bpms_m, bpms_d = interpolate_hr_pair(hr_m, hr_d, sample_interval_ms=1000)
            base_m = zip_baseline.get("matcher")
            base_d = zip_baseline.get("director")
            cross_row = {"sessionId": os.path.basename(zip_path), "trial": trial_idx, "role": "cross"}
            # CRQA (full metrics)
            crqa = crqa_features(bpms_m, bpms_d, baseline_m=base_m, baseline_d=base_d)
            if crqa:
                cross_row.update(crqa)
            # MdRQA (joint phase space)
            mdrqa = mdrqa_features(bpms_m, bpms_d, baseline_m=base_m, baseline_d=base_d)
            if mdrqa:
                cross_row.update(mdrqa)
            # DCRP (leader-follower)
            dcrp = dcrp_features(bpms_m, bpms_d, baseline_m=base_m, baseline_d=base_d)
            if dcrp:
                cross_row.update(dcrp)
            # Windowed Cross-Correlation
            wcc = windowed_xcorr(bpms_m, bpms_d)
            if wcc:
                cross_row.update(wcc)
            # Transfer Entropy (bidirectional)
            te = transfer_entropy_features(bpms_m, bpms_d)
            if te:
                cross_row.update(te)
            # Windowed CRQA (time-varying synchrony)
            wcrqa = windowed_crqa(bpms_m, bpms_d)
            if wcrqa:
                cross_row.update(wcrqa)
            # Surrogate baseline for significance testing
            surr = surrogate_crqa(bpms_m, bpms_d, baseline_m=base_m, baseline_d=base_d)
            if surr:
                cross_row.update(surr)
            if len(cross_row) > 3:  # has more than just sessionId/trial/role
                hr_stats_rows.append(cross_row)

            # Audio
            for rel in zf.namelist():
                if not rel.startswith(f"trials/{tdir}/audio/") or rel.endswith("/"):
                    continue
                try:
                    with zf.open(rel) as f:
                        audio_bytes = f.read()
                except Exception:
                    continue
                fname = os.path.basename(rel)
                # Save original audio for downstream use
                try:
                    out_audio_path = os.path.join(audio_out_dir, f"T{trial_idx:02d}_{fname}")
                    with open(out_audio_path, "wb") as af:
                        af.write(audio_bytes)
                except Exception:
                    pass
                audio_rows.append({
                    "sessionId": os.path.basename(zip_path),
                    "trial": trial_idx,
                    "filename": fname,
                    "bytes": len(audio_bytes)
                })
                # Convert non-WAV audio to WAV for downstream processing
                wav_bytes = convert_to_wav(audio_bytes, fname) if fname.endswith((".webm", ".ogg", ".opus")) else audio_bytes
                # Prosody
                pf = prosody_features(wav_bytes)
                if pf:
                    prosody_rows.append({"sessionId": os.path.basename(zip_path), "trial": trial_idx, "filename": fname, **pf})
                # ASR
                if asr_key:
                    try:
                        asr_json = asr_smallest(wav_bytes, asr_key)
                        text = asr_json.get("text") or asr_json.get("transcript") or ""
                        conf = asr_json.get("confidence") or asr_json.get("score") or ""
                        speech_rows.append({
                            "sessionId": os.path.basename(zip_path),
                            "trial": trial_idx,
                            "filename": fname,
                            "text": text,
                            "confidence": conf
                        })
                    except Exception as e:
                        speech_rows.append({
                            "sessionId": os.path.basename(zip_path),
                            "trial": trial_idx,
                            "filename": fname,
                            "text": "",
                            "confidence": "",
                            "error": str(e)[:200]
                        })

            # ── Speech pipeline (Whisper + Parselmouth + OpenSMILE + LLM eval + KG) ──
            # Collect WAV paths per role from saved audio files
            trial_wav_paths = {}  # role -> wav_path
            trial_transcripts = {}  # role -> transcript text
            for rel in zf.namelist():
                if not rel.startswith(f"trials/{tdir}/audio/") or rel.endswith("/"):
                    continue
                fname = os.path.basename(rel)
                role_audio = None
                if "director" in fname.lower():
                    role_audio = "director"
                elif "matcher" in fname.lower():
                    role_audio = "matcher"
                if role_audio:
                    saved_path = os.path.join(audio_out_dir, f"T{trial_idx:02d}_{fname}")
                    # Save WAV version for speech modules
                    if os.path.exists(saved_path):
                        if saved_path.endswith((".webm", ".ogg", ".opus")):
                            wav_path = saved_path.rsplit(".", 1)[0] + ".wav"
                            if not os.path.exists(wav_path):
                                try:
                                    with open(saved_path, "rb") as af:
                                        wav_data = convert_to_wav(af.read(), fname)
                                    with open(wav_path, "wb") as wf:
                                        wf.write(wav_data)
                                except Exception:
                                    wav_path = saved_path
                            trial_wav_paths[role_audio] = wav_path
                        else:
                            trial_wav_paths[role_audio] = saved_path

            # Per-role speech features (Whisper + Parselmouth + OpenSMILE)
            if HAS_SPEECH and openai_key and trial_wav_paths:
                for role_audio, wav_path in trial_wav_paths.items():
                    try:
                        sf_row = extract_speech_features(wav_path, role_audio, trial_idx, openai_key)
                        trial_transcripts[role_audio] = sf_row.get("transcript", "")
                        # Flatten for CSV (remove nested lists)
                        flat = {"sessionId": os.path.basename(zip_path), "trial": trial_idx, "role": role_audio}
                        for k, v in sf_row.items():
                            if k in ("words", "segments"):
                                flat[f"{k}_count"] = len(v) if isinstance(v, list) else 0
                            elif isinstance(v, (int, float, str, bool)):
                                flat[k] = v
                        speech_feat_rows.append(flat)
                    except Exception as e:
                        speech_feat_rows.append({
                            "sessionId": os.path.basename(zip_path), "trial": trial_idx,
                            "role": role_audio, "error": str(e)[:200]
                        })

                # Pair-level speech features
                if "director" in trial_wav_paths and "matcher" in trial_wav_paths:
                    try:
                        pair_f = extract_pair_features(
                            trial_wav_paths["director"], trial_wav_paths["matcher"],
                            trial_idx, openai_key)
                        pair_f["sessionId"] = os.path.basename(zip_path)
                        pair_speech_rows.append(pair_f)
                    except Exception as e:
                        pair_speech_rows.append({
                            "sessionId": os.path.basename(zip_path), "trial": trial_idx,
                            "error": str(e)[:200]
                        })

            # LLM dialogue evaluation (dialogue acts, quality, convergence)
            d_text = trial_transcripts.get("director", "")
            m_text = trial_transcripts.get("matcher", "")
            if HAS_LLM_EVAL and openai_key and (d_text or m_text):
                try:
                    llm_row = llm_evaluate_trial(d_text, m_text, trial_idx, openai_key)
                    llm_row["sessionId"] = os.path.basename(zip_path)
                    llm_eval_rows.append(llm_row)
                except Exception as e:
                    llm_eval_rows.append({
                        "sessionId": os.path.basename(zip_path), "trial": trial_idx,
                        "error": str(e)[:200]
                    })

            # Knowledge graph extraction
            if HAS_KG and openai_key and (d_text or m_text):
                try:
                    kg_gt = gt_cache.get(map_number, {}) if gt_cache else None
                    kg_row = kg_process_trial(
                        d_text, m_text, map_number,
                        gt_json=kg_gt,
                        map_image_dir=map_image_dir,
                        api_key=openai_key)
                    kg_row["sessionId"] = os.path.basename(zip_path)
                    kg_row["trial"] = trial_idx
                    kg_rows.append(kg_row)
                except Exception as e:
                    kg_rows.append({
                        "sessionId": os.path.basename(zip_path), "trial": trial_idx,
                        "mapNumber": map_number, "error": str(e)[:200]
                    })

            # ── Gaze features (from preprocessed eye CSVs) ──
            trial_label = tdir  # e.g. "T03"
            if HAS_GAZE and (gaze_data_d or gaze_data_m):
                gaze_d_trial = gaze_filter_trial(gaze_data_d, trial_label) if gaze_data_d else []
                gaze_m_trial = gaze_filter_trial(gaze_data_m, trial_label) if gaze_data_m else []

                # Per-individual gaze features
                if gaze_d_trial:
                    gf_d = extract_gaze_features(gaze_d_trial, "director", trial_label)
                    gf_d["sessionId"] = os.path.basename(zip_path)
                    gaze_feat_rows.append(gf_d)
                if gaze_m_trial:
                    gf_m = extract_gaze_features(gaze_m_trial, "matcher", trial_label)
                    gf_m["sessionId"] = os.path.basename(zip_path)
                    gaze_feat_rows.append(gf_m)

                # Cross-participant + multimodal gaze features
                if gaze_d_trial and gaze_m_trial:
                    gt_for_gaze = gt_cache.get(map_number, {})
                    gp = extract_pair_gaze_features(
                        gaze_d_trial, gaze_m_trial, trial_label,
                        strokes=strokes,
                        gt_json=gt_for_gaze if gt_for_gaze else None,
                        hr_d=hr_d, hr_m=hr_m,
                    )
                    gp["sessionId"] = os.path.basename(zip_path)
                    gaze_pair_rows.append(gp)

            # Extract trial-level metadata from events
            # Offset-correct Matcher timestamps to Director reference clock
            trial_start_t = ""
            trial_end_t = ""
            target_reached = ""
            path_confidence = ""
            director_note = ""
            tlx_director = {}
            tlx_matcher = {}
            psmm_director = {}
            psmm_matcher = {}
            for e in events:
                etype = e.get("type", "")
                e_role = e.get("role", "")
                # Skip stale events from previous sessions (localStorage backup leak)
                et = e.get("t")
                if session_t_min is not None and et is not None and et < session_t_min:
                    continue
                if session_t_max is not None and et is not None and et > session_t_max:
                    continue
                # Skip events from a different trial (some events have trialIndex in payload)
                e_ti = e.get("payload", {}).get("trialIndex")
                if isinstance(e_ti, int) and e_ti != trial_idx:
                    continue
                if etype == "trial_final_time":
                    t_val = e.get("t", "")
                    if t_val and e_role == "matcher":
                        t_val = _offset_t(t_val)
                    if not trial_end_t or (t_val and t_val > trial_end_t):
                        trial_end_t = t_val
                if etype == "draw_stroke" and not trial_start_t:
                    t_val = e.get("t", "")
                    # draw_stroke is always matcher
                    trial_start_t = _offset_t(t_val) if t_val else ""
                if etype == "trial_success":
                    p = e.get("payload", {})
                    target_reached = p.get("targetReached", "")
                    path_confidence = p.get("pathConfidence", "")
                    director_note = p.get("note", "")
                if etype == "tlx_submit":
                    role = e.get("role", "")
                    p = e.get("payload", {})
                    if role == "director":
                        tlx_director = p
                    elif role == "matcher":
                        tlx_matcher = p
                if etype == "psmm_submit":
                    role = e.get("role", "")
                    p = e.get("payload", {})
                    # PSMM payload format: {'0': {'itemNum': 1, 'value': 4, 'factor': 'task_smm'}, ..., 'trialIndex': N, 'mapNumber': M}
                    psmm_dict = {}
                    if isinstance(p, dict):
                        for k, v in p.items():
                            if isinstance(v, dict) and 'itemNum' in v and 'value' in v:
                                psmm_dict[str(v['itemNum'])] = v['value']
                            elif isinstance(v, (int, float)) and k not in ('trialIndex', 'mapNumber'):
                                # Fallback: direct key=value
                                psmm_dict[str(k)] = v
                    if role == "director":
                        psmm_director = psmm_dict
                    elif role == "matcher":
                        psmm_matcher = psmm_dict

            manifest_rows.append({
                "sessionId": os.path.basename(zip_path),
                "trial": trial_idx,
                "mapNumber": map_number,
                "clock_offset_ms": round(matcher_offset_ms, 1),
                "ref_clock": "director",
                "trial_start_ms": trial_start_t,
                "trial_start_iso": epoch_to_iso(trial_start_t),
                "trial_end_ms": trial_end_t,
                "trial_end_iso": epoch_to_iso(trial_end_t),
                "strokes": len(strokes),
                "strokePoints": sum(len(s.get("points", [])) for s in strokes),
                "hr_matcher": len(hr_m),
                "hr_director": len(hr_d),
                "audio_count": len([r for r in audio_rows if r["trial"] == trial_idx]),
                "coverage_gt": coverage_gt,
                "coverage_pred": coverage_pred,
                "target_reached": target_reached,
                "path_confidence": path_confidence,
                "director_note": director_note,
                "tlx_mental_d": tlx_director.get("mental", ""),
                "tlx_physical_d": tlx_director.get("physical", ""),
                "tlx_temporal_d": tlx_director.get("temporal", ""),
                "tlx_performance_d": tlx_director.get("performance", ""),
                "tlx_effort_d": tlx_director.get("effort", ""),
                "tlx_frustration_d": tlx_director.get("frustration", ""),
                "tlx_mental_m": tlx_matcher.get("mental", ""),
                "tlx_physical_m": tlx_matcher.get("physical", ""),
                "tlx_temporal_m": tlx_matcher.get("temporal", ""),
                "tlx_performance_m": tlx_matcher.get("performance", ""),
                "tlx_effort_m": tlx_matcher.get("effort", ""),
                "tlx_frustration_m": tlx_matcher.get("frustration", ""),
                # PSMM (perceived shared mental model). Items 1-4 = task SMM (route/landmarks/obstacles/position)
                # Items 5-8 = team SMM (anticipation/role/communication/resolution)
                "psmm_route_d": psmm_director.get("1", psmm_director.get(1, "")),
                "psmm_landmarks_d": psmm_director.get("2", psmm_director.get(2, "")),
                "psmm_obstacles_d": psmm_director.get("3", psmm_director.get(3, "")),
                "psmm_position_d": psmm_director.get("4", psmm_director.get(4, "")),
                "psmm_anticipation_d": psmm_director.get("5", psmm_director.get(5, "")),
                "psmm_role_d": psmm_director.get("6", psmm_director.get(6, "")),
                "psmm_communication_d": psmm_director.get("7", psmm_director.get(7, "")),
                "psmm_resolution_d": psmm_director.get("8", psmm_director.get(8, "")),
                "psmm_route_m": psmm_matcher.get("1", psmm_matcher.get(1, "")),
                "psmm_landmarks_m": psmm_matcher.get("2", psmm_matcher.get(2, "")),
                "psmm_obstacles_m": psmm_matcher.get("3", psmm_matcher.get(3, "")),
                "psmm_position_m": psmm_matcher.get("4", psmm_matcher.get(4, "")),
                "psmm_anticipation_m": psmm_matcher.get("5", psmm_matcher.get(5, "")),
                "psmm_role_m": psmm_matcher.get("6", psmm_matcher.get(6, "")),
                "psmm_communication_m": psmm_matcher.get("7", psmm_matcher.get(7, "")),
                "psmm_resolution_m": psmm_matcher.get("8", psmm_matcher.get(8, "")),
            })

            # Time-series correctness per stroke
            current_mask = np.zeros_like(pred_mask)
            step = 0
            for sraw in sorted(strokes, key=lambda s: s.get("t") or 0):
                # draw this stroke onto current_mask
                img = Image.fromarray((current_mask == 0).astype(np.uint8) * 255, mode="L")
                draw = ImageDraw.Draw(img)
                erase = sraw.get("mode", "draw") == "erase"
                stroke_draw(draw, sraw, erase)
                current_mask = (np.array(img) == 0).astype(np.uint8)

                step += 1
                if step % 5 != 0 and step != len(strokes):
                    continue  # downsample to every 5th stroke, keep final
                m_ts = binary_metrics(gt_mask, current_mask)
                cd_ts = chamfer(gt_mask, current_mask)
                bf_ts = boundary_f(gt_mask, current_mask, 2)
                t_val = _offset_t(sraw.get("t", ""))
                ts_rows.append({
                    "sessionId": os.path.basename(zip_path),
                    "trial": trial_idx,
                    "mapNumber": map_number,
                    "step": step,
                    "t_unix_ms": t_val,
                    "t_iso": epoch_to_iso(t_val),
                    "iou": m_ts["iou"],
                    "f1": m_ts["f1"],
                    "dice": m_ts["dice"],
                    "precision": m_ts["precision"],
                    "recall": m_ts["recall"],
                    "chamfer": cd_ts,
                    "boundary_f1": bf_ts["f1"],
                    "boundary_p": bf_ts["precision"],
                    "boundary_r": bf_ts["recall"],
                    "coverage_pred": float(current_mask.mean()),
                })

        # Write CSVs
        out = os.path.join(out_dir, "metrics.csv"); csv_write(metrics_rows, out)
        out = os.path.join(out_dir, "strokes.csv"); csv_write(strokes_rows, out)
        out = os.path.join(out_dir, "hr_matcher.csv"); csv_write(hr_matcher_rows, out)
        out = os.path.join(out_dir, "hr_director.csv"); csv_write(hr_director_rows, out)
        out = os.path.join(out_dir, "audio_manifest.csv"); csv_write(audio_rows, out)
        out = os.path.join(out_dir, "manifest.csv"); csv_write(manifest_rows, out)
        if prosody_rows: csv_write(prosody_rows, os.path.join(out_dir, "prosody.csv"))
        if speech_rows: csv_write(speech_rows, os.path.join(out_dir, "speech.csv"))
        if hr_stats_rows: csv_write(hr_stats_rows, os.path.join(out_dir, "hr_stats.csv"))
        if ts_rows: csv_write(ts_rows, os.path.join(out_dir, "time_series_metrics.csv"))
        if speech_feat_rows: csv_write(speech_feat_rows, os.path.join(out_dir, "speech_features.csv"))
        if pair_speech_rows: csv_write(pair_speech_rows, os.path.join(out_dir, "pair_speech.csv"))
        if llm_eval_rows: csv_write(llm_eval_rows, os.path.join(out_dir, "llm_eval.csv"))
        if kg_rows: csv_write(kg_rows, os.path.join(out_dir, "knowledge_graph.csv"))
        if gaze_feat_rows: csv_write(gaze_feat_rows, os.path.join(out_dir, "gaze_features.csv"))
        if gaze_pair_rows: csv_write(gaze_pair_rows, os.path.join(out_dir, "gaze_pair.csv"))
        if drawing_feat_rows: csv_write(drawing_feat_rows, os.path.join(out_dir, "drawing_features.csv"))

        # ML-ready joined table: one row per trial, all features prefixed by modality
        ml_rows = build_ml_table(
            metrics_rows, hr_stats_rows, drawing_feat_rows,
            gaze_feat_rows, gaze_pair_rows, speech_feat_rows,
            pair_speech_rows, llm_eval_rows, manifest_rows,
            prosody_rows=prosody_rows, kg_rows=kg_rows,
        )
        if ml_rows:
            csv_write(ml_rows, os.path.join(out_dir, "ml_ready.csv"))
            n_cols = len(ml_rows[0]) if ml_rows else 0
            print(f"[ml_ready] {len(ml_rows)} trial rows × {n_cols} columns → ml_ready.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Session ZIP path")
    ap.add_argument("--gt-dir", default="Ground Truth Maps", help="GT JSON directory (gt_#.json)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--smallest-key", default=None, help="Smallest Pulse API key (optional for ASR)")
    ap.add_argument("--openai-key", default=None, help="OpenAI API key (for Whisper, LLM eval, KG)")
    ap.add_argument("--map-image-dir", default=None,
                     help="Directory containing map images (map{N}f.gif) for KG vision extraction")
    ap.add_argument("--eye-director", default=None,
                     help="Preprocessed eye CSV for Director (from preprocess_eye.py)")
    ap.add_argument("--eye-matcher", default=None,
                     help="Preprocessed eye CSV for Matcher (from preprocess_eye.py)")
    args = ap.parse_args()
    process_zip(args.zip, args.gt_dir, args.out,
                asr_key=args.smallest_key or os.getenv("SMALLEST_AI_KEY"),
                openai_key=args.openai_key or os.getenv("OPENAI_API_KEY"),
                map_image_dir=args.map_image_dir,
                eye_csv_director=args.eye_director,
                eye_csv_matcher=args.eye_matcher)


if __name__ == "__main__":
    main()
