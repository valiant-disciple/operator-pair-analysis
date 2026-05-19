#!/usr/bin/env python3
"""
Eye-tracker data preprocessor: Aurora (iMotions CSV) and SmartEye Pro 10 (.log TSV)
→ unified gaze CSV with AOI labels and trial alignment.

Usage:
  python scripts/preprocess_eye.py \
    --eye-file path/to/eye_data.csv \
    --format aurora \
    --role director \
    --zip path/to/session.zip \
    --out eye_preprocessed.csv
"""

import argparse
import datetime
import json
import math
import os
import re
import zipfile

AOI_CONFIG = {
    "director": {
        # Map rect inferred from 488K pooled gaze samples (40 sessions).
        # Aspect 0.722 = exact match for 651×900 GIF (0.723), confirming this IS the map.
        "map": {"x1": 640, "y1": 166, "x2": 1120, "y2": 831},
        "timer": {"x1": 613, "y1": 8, "x2": 735, "y2": 65},
        "toolbar": {"x1": 236, "y1": 0, "x2": 1018, "y2": 74},
    },
    "matcher": {
        # Same inference; matcher canvas shifted ~80px left of director.
        "map": {"x1": 560, "y1": 166, "x2": 1040, "y2": 831},
        "timer": {"x1": 613, "y1": 8, "x2": 735, "y2": 65},
        "toolbar": {"x1": 236, "y1": 0, "x2": 1365, "y2": 74},
    },
}

OUTPUT_COLUMNS = [
    "t_unix_ms", "t_iso", "trial", "gaze_x", "gaze_y", "aoi",
    "pupil_left", "pupil_right", "head_pitch", "head_yaw", "head_roll",
    "fixation_idx", "fixation_x", "fixation_y", "fixation_duration",
    "saccade_idx", "saccade_amplitude", "saccade_peak_velocity",
    "saccade_direction", "gaze_velocity", "blink",
    "eyelid_left", "eyelid_right", "role", "source",
]

WINDOWS_EPOCH_OFFSET_MS = 11644473600000


def epoch_to_iso(t) -> str:
    if t is None or t == "":
        return ""
    try:
        return datetime.datetime.fromtimestamp(
            int(t) / 1000, tz=datetime.timezone.utc
        ).isoformat()
    except (ValueError, TypeError, OSError):
        return ""


def csv_write(rows: list, path: str):
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
                if any(c in s for c in [",", '"', "\n"]):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(",".join(vals) + "\n")


def _safe_float(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(v):
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def classify_aoi(gaze_x, gaze_y, role: str) -> str:
    if gaze_x is None or gaze_y is None:
        return "missing"
    aois = AOI_CONFIG.get(role, AOI_CONFIG["director"])
    t = aois["timer"]
    if t["x1"] <= gaze_x <= t["x2"] and t["y1"] <= gaze_y <= t["y2"]:
        return "timer"
    m = aois["map"]
    if m["x1"] <= gaze_x <= m["x2"] and m["y1"] <= gaze_y <= m["y2"]:
        return "map"
    tb = aois["toolbar"]
    if tb["x1"] <= gaze_x <= tb["x2"] and tb["y1"] <= gaze_y <= tb["y2"]:
        return "toolbar"
    return "other"


# ---------------------------------------------------------------------------
# Trial boundaries from session ZIP
# ---------------------------------------------------------------------------

def extract_trial_boundaries(zip_path: str) -> list:
    """Return sorted list of (trial_label, start_ms, end_ms)."""
    trials = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trial_dirs = sorted(
                {p.split("/")[1] for p in zf.namelist()
                 if p.startswith("trials/") and len(p.split("/")) > 2}
            )
            for tdir in trial_dirs:
                events_path = f"trials/{tdir}/events.json"
                try:
                    with zf.open(events_path) as ef:
                        events = json.loads(ef.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    continue
                if not events:
                    continue

                start_ms = None
                end_ms = None
                for e in events:
                    et = e.get("t")
                    if et is None:
                        continue
                    t_val = _safe_int(et)
                    if t_val is None:
                        continue
                    if e.get("type") == "draw_stroke" and start_ms is None:
                        start_ms = t_val
                    if e.get("type") == "trial_final_time":
                        end_ms = t_val

                if start_ms is None and events:
                    for e in events:
                        t_val = _safe_int(e.get("t"))
                        if t_val is not None:
                            start_ms = t_val
                            break
                if end_ms is None and events:
                    for e in reversed(events):
                        t_val = _safe_int(e.get("t"))
                        if t_val is not None:
                            end_ms = t_val
                            break
                if start_ms is not None and end_ms is not None:
                    trials.append((tdir, start_ms, end_ms))
    except (zipfile.BadZipFile, FileNotFoundError) as exc:
        print(f"Warning: could not read ZIP for trial boundaries: {exc}")
    return sorted(trials, key=lambda x: x[1])


def extract_flash_timestamps(zip_path: str) -> list:
    """Return list of {trial_label, flash_ts} from sync_flash events in events.json."""
    flashes = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trial_dirs = sorted(
                {p.split("/")[1] for p in zf.namelist()
                 if p.startswith("trials/") and len(p.split("/")) > 2}
            )
            for tdir in trial_dirs:
                events_path = f"trials/{tdir}/events.json"
                try:
                    with zf.open(events_path) as ef:
                        events = json.loads(ef.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    continue
                for e in events:
                    if e.get("type") == "sync_flash":
                        payload = e.get("payload", {})
                        flash_ts = _safe_int(payload.get("flashTs") or e.get("t"))
                        if flash_ts:
                            flashes.append({"trial": tdir, "flash_ts": flash_ts})
    except (zipfile.BadZipFile, FileNotFoundError):
        pass
    return flashes


def extract_clock_offset(zip_path: str) -> dict:
    """Return clock_offset event data from events.json if present."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trial_dirs = sorted(
                {p.split("/")[1] for p in zf.namelist()
                 if p.startswith("trials/") and len(p.split("/")) > 2}
            )
            # clock_offset is typically in the first trial's events
            for tdir in trial_dirs:
                events_path = f"trials/{tdir}/events.json"
                try:
                    with zf.open(events_path) as ef:
                        events = json.loads(ef.read().decode("utf-8"))
                except (KeyError, json.JSONDecodeError):
                    continue
                for e in events:
                    if e.get("type") == "clock_offset":
                        return e.get("payload", {})
            # Also check top-level events.json
            for name in zf.namelist():
                if name == "events.json" or name.endswith("/events.json"):
                    try:
                        with zf.open(name) as ef:
                            events = json.loads(ef.read().decode("utf-8"))
                        for e in events:
                            if e.get("type") == "clock_offset":
                                return e.get("payload", {})
                    except (KeyError, json.JSONDecodeError):
                        continue
    except (zipfile.BadZipFile, FileNotFoundError):
        pass
    return {}


def detect_flash_in_pupil(rows: list, flash_ts_unix_ms: int,
                          search_window_ms: int = 2000) -> int:
    """
    Detect the pupil constriction onset caused by a sync flash.

    The pupillary light reflex fires AFTER the flash (physiological latency
    ~150-500ms). We therefore search only in a forward window starting 100ms
    after the flash (to skip the flash itself) and ending at flash + 800ms
    (upper bound of normal reflex latency). This prevents pre-flash blinks
    from being mistakenly chosen as the constriction response.

    Returns the eye-tracker timestamp (t_unix_ms) of the detected constriction
    onset, or 0 if not detected.
    """
    # Search ONLY after the flash — pupil reflex is always forward in time.
    # [+100ms, +800ms] captures the physiologically plausible window.
    # Use the broader 2000ms window only as a fallback if the tight window
    # yields too few samples (e.g., participant blinked during flash).
    window_start = flash_ts_unix_ms + 100
    window_end = flash_ts_unix_ms + 800
    samples = []
    for r in rows:
        t = _safe_int(r.get("t_unix_ms"))
        if t is None or t < window_start or t > window_end:
            continue
        pl = _safe_float(r.get("pupil_left"))
        pr = _safe_float(r.get("pupil_right"))
        vals = [v for v in [pl, pr] if v is not None and v > 0]
        if vals:
            samples.append((t, sum(vals) / len(vals)))

    # Fallback: widen to ±search_window_ms if too few samples (e.g., blink)
    if len(samples) < 5:
        window_start = flash_ts_unix_ms - search_window_ms
        window_end = flash_ts_unix_ms + search_window_ms
        samples = []
        for r in rows:
            t = _safe_int(r.get("t_unix_ms"))
            if t is None or t < window_start or t > window_end:
                continue
            pl = _safe_float(r.get("pupil_left"))
            pr = _safe_float(r.get("pupil_right"))
            vals = [v for v in [pl, pr] if v is not None and v > 0]
            if vals:
                samples.append((t, sum(vals) / len(vals)))

    if len(samples) < 10:
        return 0

    # Find the sharpest pupil diameter drop (constriction onset).
    # Returns the timestamp of the sample BEFORE the steepest drop.
    best_drop = 0
    best_t = 0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        if dt <= 0:
            continue
        dp = samples[i][1] - samples[i - 1][1]
        rate = dp / dt  # mm/ms — negative = constriction
        if rate < best_drop:
            best_drop = rate
            best_t = samples[i - 1][0]

    return best_t


# Typical pupillary light reflex onset latency in ms.
# This is subtracted from the raw offset so the correction maps the
# trial start (flashTs) to t=0 in eye tracker time, not the constriction onset.
_PUPIL_REFLEX_LATENCY_MS = 250


def find_coarse_offset_by_flash_pattern(rows: list,
                                        flash_timestamps_ms: list,
                                        tolerance_ms: int = 8000) -> "int | None":
    """
    Find the clock offset between eye tracker and browser when the offset is
    too large for per-trial window search (e.g., eye tracker machine clock is
    minutes ahead or behind the browser machine).

    Strategy:
      1. Build a downsampled pupil signal (~10Hz) from the full recording.
      2. Find all timepoints with a significant pupil drop over the next 600ms
         (candidates for flash-induced constrictions).
      3. Keep the top N_CANDIDATES by drop magnitude.
      4. For each candidate as a potential first-flash match, check whether
         the remaining flashes can be found at the expected inter-trial offsets.
      5. The candidate set with the most matches gives the coarse offset.

    Returns estimated offset (eye_tracker_ms - browser_ms), or None if the
    pattern cannot be identified with sufficient confidence.
    """
    n_flashes = len(flash_timestamps_ms)
    if n_flashes < 3:
        return None

    # Build ~10Hz pupil time series (skip samples < 80ms apart)
    series = []
    prev_t = -99999
    for r in rows:
        t = _safe_int(r.get("t_unix_ms"))
        if t is None or t - prev_t < 80:
            continue
        pl = _safe_float(r.get("pupil_left"))
        pr = _safe_float(r.get("pupil_right"))
        vals = [v for v in [pl, pr] if v is not None and v > 0]
        if vals:
            series.append((t, sum(vals) / len(vals)))
            prev_t = t

    if len(series) < n_flashes * 15:
        return None

    # For each point, compute the mean pupil 400-700ms later.
    # A large positive "drop" = constriction = candidate flash response.
    n = len(series)
    drop_events = []
    j = 0
    for i in range(n):
        t0, p0 = series[i]
        # advance j to first sample >= t0+400ms
        while j < n - 1 and series[j][0] < t0 + 400:
            j += 1
        # collect samples in [t0+400, t0+700]
        laters = [p for t, p in series[j:] if t <= t0 + 700]
        if len(laters) < 2:
            continue
        nadir = sum(laters) / len(laters)
        drop = p0 - nadir
        if drop > 0.15:  # >0.15mm = meaningful constriction
            drop_events.append((t0, drop))

    if len(drop_events) < n_flashes:
        return None

    # Keep top candidates by drop magnitude (at most 5× the number of flashes)
    drop_events.sort(key=lambda x: -x[1])
    candidates = sorted(t for t, _ in drop_events[: n_flashes * 5])

    # Inter-trial intervals from browser timestamps
    intervals = [flash_timestamps_ms[k + 1] - flash_timestamps_ms[k]
                 for k in range(n_flashes - 1)]

    best_score = 0
    best_raw_offsets: "list[int]" = []

    for cand in candidates:
        # Treat cand as the eye-tracker timestamp of the first flash.
        # Build expected eye-tracker timestamps for all subsequent flashes.
        score = 1
        matched_et: "list[int | None]" = [cand]
        cumulative = 0
        for interval in intervals:
            cumulative += interval
            expected_et = cand + cumulative
            hit = next(
                (c for c in candidates if abs(c - expected_et) <= tolerance_ms),
                None,
            )
            matched_et.append(hit)
            if hit is not None:
                score += 1

        if score > best_score:
            best_score = score
            raw_offsets = []
            for k, met in enumerate(matched_et):
                if met is not None and k < n_flashes:
                    raw_offsets.append(met - flash_timestamps_ms[k])
            best_raw_offsets = raw_offsets

    min_required = max(3, (n_flashes * 2) // 3)
    if best_score >= min_required and best_raw_offsets:
        coarse = int(sum(best_raw_offsets) / len(best_raw_offsets))
        print(f"  Pattern match: {best_score}/{n_flashes} flashes identified, "
              f"coarse raw offset={coarse:+.0f}ms "
              f"(latency-corrected: {coarse - _PUPIL_REFLEX_LATENCY_MS:+.0f}ms)")
        return coarse
    return None


def extract_ttl_flash_timestamps(rows: list, ttl_column: str) -> list:
    """
    Extract flash event timestamps from a hardware TTL/photodiode column.

    Detects rising edges (0→non-zero, or below-threshold→above-threshold) in
    the named column. Each rising edge is one flash event.

    Returns list of t_unix_ms values for each detected flash onset.
    This is the gold-standard method: no physiological latency, no pattern
    matching needed, precision = 1 eye tracker frame (~4ms at 250Hz).
    """
    events = []
    prev_val = 0.0
    threshold = 0.5  # TTL is typically 0/1 or 0/5V normalised to 0/1
    for r in rows:
        t = _safe_int(r.get("t_unix_ms"))
        if t is None:
            continue
        raw = r.get(ttl_column, "")
        val = _safe_float(raw)
        if val is None:
            val = 1.0 if str(raw).strip() not in ("", "0", "0.0", "false", "False") else 0.0
        if prev_val <= threshold < val:  # rising edge
            events.append(t)
        prev_val = val
    return events


def assign_trial(t_unix_ms, trial_boundaries: list) -> str:
    if t_unix_ms is None:
        return "no_trial"
    for label, start, end in trial_boundaries:
        if start <= t_unix_ms <= end:
            return label
    return "no_trial"


# ---------------------------------------------------------------------------
# Aurora (iMotions) CSV parser
# ---------------------------------------------------------------------------

def _parse_csv_line(line: str) -> list:
    """Minimal CSV field splitter that handles quoted fields."""
    fields = []
    current = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == '"' and i + 1 < len(line) and line[i + 1] == '"':
                current.append('"')
                i += 2
                continue
            elif ch == '"':
                in_quotes = False
            else:
                current.append(ch)
        else:
            if ch == '"':
                in_quotes = True
            elif ch == ',':
                fields.append("".join(current))
                current = []
            else:
                current.append(ch)
        i += 1
    fields.append("".join(current))
    return fields


def _col(header_map: dict, row: list, name: str, default=""):
    idx = header_map.get(name)
    if idx is None or idx >= len(row):
        return default
    v = row[idx]
    return v if v != "" else default


def parse_aurora(eye_path: str, role: str, trial_boundaries: list) -> list:
    recording_unix_s = None
    data_header = None
    rows_out = []

    with open(eye_path, "r", encoding="utf-8", errors="replace") as f:
        found_data_marker = False
        for raw_line in f:
            line = raw_line.rstrip("\n\r")

            if line.startswith("#"):
                m = re.search(r"Unix time:\s*(\d+)", line)
                if m and recording_unix_s is None:
                    recording_unix_s = int(m.group(1))
                if line.startswith("#DATA"):
                    found_data_marker = True
                continue

            if found_data_marker and data_header is None:
                data_header = _parse_csv_line(line)
                hmap = {name: i for i, name in enumerate(data_header)}
                continue

            if data_header is None:
                continue

            fields = _parse_csv_line(line)
            if not fields or len(fields) < 2:
                continue

            ts_str = _col(hmap, fields, "Timestamp")
            ts_ms = _safe_float(ts_str)
            if ts_ms is None:
                continue

            if recording_unix_s is None:
                continue
            t_unix_ms = int(recording_unix_s * 1000 + ts_ms)

            gx = _safe_float(_col(hmap, fields, "Gaze X"))
            gy = _safe_float(_col(hmap, fields, "Gaze Y"))
            if gx is None or gy is None:
                gx = _safe_float(_col(hmap, fields, "Interpolated Gaze X"))
                gy = _safe_float(_col(hmap, fields, "Interpolated Gaze Y"))

            aoi = classify_aoi(gx, gy, role)
            trial = assign_trial(t_unix_ms, trial_boundaries)

            eyelid_l_str = _col(hmap, fields, "ET_EyelidOpeningLeft")
            eyelid_r_str = _col(hmap, fields, "ET_EyelidOpeningRight")

            # Infer blink from eyelid opening
            el = _safe_float(eyelid_l_str)
            er = _safe_float(eyelid_r_str)
            blink_detected = ""
            if el is not None and er is not None:
                if el < 0.2 and er < 0.2:
                    blink_detected = "1"
            elif el is not None and el < 0.2:
                blink_detected = "1"
            elif er is not None and er < 0.2:
                blink_detected = "1"

            rows_out.append({
                "t_unix_ms": t_unix_ms,
                "t_iso": epoch_to_iso(t_unix_ms),
                "trial": trial,
                "gaze_x": gx if gx is not None else "",
                "gaze_y": gy if gy is not None else "",
                "aoi": aoi,
                "pupil_left": _col(hmap, fields, "ET_PupilLeft"),
                "pupil_right": _col(hmap, fields, "ET_PupilRight"),
                "head_pitch": _col(hmap, fields, "ET_HeadRotationPitch"),
                "head_yaw": _col(hmap, fields, "ET_HeadRotationYaw"),
                "head_roll": _col(hmap, fields, "ET_HeadRotationRoll"),
                "fixation_idx": _col(hmap, fields, "Fixation Index"),
                "fixation_x": _col(hmap, fields, "Fixation X"),
                "fixation_y": _col(hmap, fields, "Fixation Y"),
                "fixation_duration": _col(hmap, fields, "Fixation Duration"),
                "saccade_idx": _col(hmap, fields, "Saccade Index"),
                "saccade_amplitude": _col(hmap, fields, "Saccade Amplitude"),
                "saccade_peak_velocity": _col(hmap, fields, "Saccade Peak Velocity"),
                "saccade_direction": _col(hmap, fields, "Saccade Direction"),
                "gaze_velocity": _col(hmap, fields, "Gaze Velocity"),
                "blink": blink_detected,
                "eyelid_left": eyelid_l_str,
                "eyelid_right": eyelid_r_str,
                "role": role,
                "source": "aurora",
                # TTL/photodiode columns preserved as-is for flash detection
                "_ttl_raw": _col(hmap, fields, "StimFrame")
                            or _col(hmap, fields, "TTL")
                            or _col(hmap, fields, "EventSignal")
                            or _col(hmap, fields, "KeyboardSignal"),
            })

    return rows_out


# ---------------------------------------------------------------------------
# SmartEye Pro 10 .log (TSV) parser
# ---------------------------------------------------------------------------

def parse_smarteye(eye_path: str, role: str, trial_boundaries: list,
                   screen_name: str = "Screen2") -> list:
    rows_out = []
    seen_screen_names = set()

    with open(eye_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\n\r")
        headers = header_line.split("\t")
        hmap = {name: i for i, name in enumerate(headers)}

        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            fields = line.split("\t")

            obj_name = _col(hmap, fields,
                            "FilteredClosestWorldIntersection.objectName")
            if not obj_name or obj_name == "0":
                obj_name = _col(hmap, fields,
                                "ClosestWorldIntersection.objectName")
            if obj_name:
                seen_screen_names.add(obj_name)
            if screen_name != "auto" and obj_name != screen_name:
                continue

            rtc_str = _col(hmap, fields, "RealTimeClock")
            rtc = _safe_int(rtc_str)
            if rtc is None or rtc == 0:
                continue
            t_unix_ms = rtc // 10000 - WINDOWS_EPOCH_OFFSET_MS

            gx = _safe_float(
                _col(hmap, fields,
                     "FilteredClosestWorldIntersection.objectPoint.x"))
            gy = _safe_float(
                _col(hmap, fields,
                     "FilteredClosestWorldIntersection.objectPoint.y"))
            if gx is None or gy is None or (gx == 0 and gy == 0):
                gx_fb = _safe_float(
                    _col(hmap, fields,
                         "ClosestWorldIntersection.objectPoint.x"))
                gy_fb = _safe_float(
                    _col(hmap, fields,
                         "ClosestWorldIntersection.objectPoint.y"))
                if gx_fb is not None and gy_fb is not None:
                    gx, gy = gx_fb, gy_fb

            aoi = classify_aoi(gx, gy, role)
            trial = assign_trial(t_unix_ms, trial_boundaries)

            # Pupil: prefer filtered, fall back to raw, then combined
            # SmartEye reports in meters — convert to mm for consistency with Aurora
            pupil_left = (_col(hmap, fields, "FilteredLeftPupilDiameter")
                          or _col(hmap, fields, "LeftPupilDiameter"))
            pupil_right = (_col(hmap, fields, "FilteredRightPupilDiameter")
                           or _col(hmap, fields, "RightPupilDiameter"))
            if not pupil_left and not pupil_right:
                pd = (_col(hmap, fields, "FilteredPupilDiameter")
                      or _col(hmap, fields, "PupilDiameter"))
                pupil_left = pd
                pupil_right = pd
            # Convert meters to millimeters
            pl_f = _safe_float(pupil_left)
            pr_f = _safe_float(pupil_right)
            if pl_f is not None and 0 < pl_f < 0.1:  # clearly in meters
                pupil_left = str(pl_f * 1000)
            if pr_f is not None and 0 < pr_f < 0.1:
                pupil_right = str(pr_f * 1000)

            blink_val = _col(hmap, fields, "Blink")

            # Eyelid opening for blink detection fallback
            eyelid_left = _col(hmap, fields, "LeftEyelidOpening")
            eyelid_right = _col(hmap, fields, "RightEyelidOpening")

            rows_out.append({
                "t_unix_ms": t_unix_ms,
                "t_iso": epoch_to_iso(t_unix_ms),
                "trial": trial,
                "gaze_x": gx if gx is not None else "",
                "gaze_y": gy if gy is not None else "",
                "aoi": aoi,
                "pupil_left": pupil_left,
                "pupil_right": pupil_right,
                "head_pitch": _col(hmap, fields, "HeadPitch"),
                "head_yaw": _col(hmap, fields, "HeadHeading"),
                "head_roll": _col(hmap, fields, "HeadRoll"),
                "fixation_idx": _col(hmap, fields, "Fixation"),
                "fixation_x": "",
                "fixation_y": "",
                "fixation_duration": "",
                "saccade_idx": _col(hmap, fields, "Saccade"),
                "saccade_amplitude": "",
                "saccade_peak_velocity": "",
                "saccade_direction": "",
                "gaze_velocity": "",
                "blink": blink_val,
                "eyelid_left": eyelid_left,
                "eyelid_right": eyelid_right,
                "role": role,
                "source": "smarteye",
                # TTL/photodiode columns preserved for flash detection
                "_ttl_raw": _col(hmap, fields, "TTL")
                            or _col(hmap, fields, "StimulusIndex")
                            or _col(hmap, fields, "TriggerIn"),
            })

    return rows_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Preprocess eye-tracker data to unified CSV")
    ap.add_argument("--eye-file", required=True, help="Eye tracker data file")
    ap.add_argument("--format", required=True, choices=["aurora", "smarteye"],
                    help="Input format: aurora or smarteye")
    ap.add_argument("--role", required=True, choices=["director", "matcher"],
                    help="Participant role")
    ap.add_argument("--zip", default=None,
                    help="Session ZIP for trial boundaries and sync flash detection")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--apply-offset", type=float, default=None,
                    help="Clock offset in ms (eye_tracker - frontend); subtracted from eye tracker timestamps")
    ap.add_argument("--cross-machine", action="store_true",
                    help="Eye tracker is on a different machine whose clock may be minutes off. "
                         "Forces pattern-matching alignment before per-trial fine-tuning. "
                         "Use this for the Director role if the eye tracker PC clock is not NTP-synced.")
    ap.add_argument("--ttl-column", default=None,
                    help="Name of the TTL/photodiode column in the eye tracker CSV "
                         "(e.g. 'StimFrame', 'TTL', 'TriggerIn'). When provided, rising edges "
                         "in this column are used for flash alignment instead of pupil constriction. "
                         "This is the gold-standard method: ~4ms precision, no clock assumptions.")
    ap.add_argument("--screen-name", default="Screen2",
                    help="SmartEye screen object name to filter gaze rows (default: Screen2). "
                         "Set to 'auto' to accept any screen name.")
    ap.add_argument("--lsl-markers-file", default=None,
                    help="Path to flash_markers.csv produced by lsl_flash_receiver.py. "
                         "Use this for the Director role when LSL sync is set up. "
                         "Provides ~3ms precision without any clock assumptions or pupil detection. "
                         "Takes priority over --cross-machine and pupil constriction methods.")
    args = ap.parse_args()

    trial_boundaries = []
    flash_events = []
    clock_offset_info = {}
    if args.zip:
        trial_boundaries = extract_trial_boundaries(args.zip)
        flash_events = extract_flash_timestamps(args.zip)
        clock_offset_info = extract_clock_offset(args.zip)
        print(f"Found {len(trial_boundaries)} trial(s) in ZIP")
        for label, s, e in trial_boundaries:
            print(f"  {label}: {s} – {e}")
        if flash_events:
            print(f"Found {len(flash_events)} sync flash event(s)")
        if clock_offset_info:
            print(f"Clock offset (software): {clock_offset_info.get('offsetMs', '?')}ms "
                  f"(RTT: {clock_offset_info.get('rttMs', '?')}ms)")

    if args.format == "aurora":
        rows = parse_aurora(args.eye_file, args.role, trial_boundaries)
    else:
        rows = parse_smarteye(args.eye_file, args.role, trial_boundaries,
                              screen_name=args.screen_name)

    if not rows:
        print("WARNING: no gaze samples produced.")
        if args.format == "smarteye":
            print(f"  Screen name filter: '{args.screen_name}'")
            if hasattr(parse_smarteye, '__code__'):
                # Re-read to discover available screen names
                seen = set()
                with open(args.eye_file, "r", encoding="utf-8", errors="replace") as _f:
                    hdr = _f.readline().rstrip("\n\r").split("\t")
                    _hm = {n: i for i, n in enumerate(hdr)}
                    for _ln in _f:
                        _fs = _ln.rstrip("\n\r").split("\t")
                        _on = _col(_hm, _fs, "FilteredClosestWorldIntersection.objectName") or _col(_hm, _fs, "ClosestWorldIntersection.objectName")
                        if _on and _on != "0":
                            seen.add(_on)
                        if len(seen) >= 10:
                            break
                if seen:
                    print(f"  Available screen names in file: {sorted(seen)}")
                    print(f"  Try: --screen-name {sorted(seen)[0]}")

    # ── Flash-based clock offset detection ──
    # Priority order:
    #   0. LSL markers file (--lsl-markers-file): ~3ms, cross-machine, no clock assumptions
    #   1. Hardware TTL/photodiode (--ttl-column): ~4ms, no clock assumptions
    #   2. Pattern-matching (--cross-machine): ~±50ms for large cross-machine offset
    #   3. Per-trial pupil constriction (default): ~±30ms, same/NTP-synced machines

    # ── Method 0: LSL marker file ──
    if flash_events and args.lsl_markers_file:
        import csv as _csv
        lsl_map: "dict[str, int]" = {}  # trial label → eye_tracker_unix_ms
        try:
            with open(args.lsl_markers_file, newline="", encoding="utf-8") as lf:
                reader = _csv.DictReader(lf)
                for row in reader:
                    trial_key = str(row.get("trial", "")).strip()
                    et_ms = _safe_int(row.get("eye_tracker_unix_ms"))
                    if trial_key and et_ms:
                        lsl_map[trial_key] = et_ms
        except FileNotFoundError:
            print(f"WARNING: --lsl-markers-file not found: {args.lsl_markers_file}")
            lsl_map = {}

        if lsl_map:
            lsl_offsets = []
            for fe in flash_events:
                trial_key = fe["trial"]
                # trial key in flash_events is like "T01"; LSL file stores trialIndex int
                # try both formats
                et_ms = lsl_map.get(trial_key) or lsl_map.get(trial_key.lstrip("T").lstrip("0") or "0")
                if et_ms:
                    offset = et_ms - fe["flash_ts"]
                    lsl_offsets.append(offset)
                    print(f"  LSL sync ({trial_key}): browser_ts={fe['flash_ts']}, "
                          f"eye_tracker_ts={et_ms}, offset={offset:+.0f}ms")
                else:
                    print(f"  LSL sync ({trial_key}): no marker in file")

            if lsl_offsets:
                mean_off = sum(lsl_offsets) / len(lsl_offsets)
                std_off = (sum((x - mean_off) ** 2 for x in lsl_offsets) / len(lsl_offsets)) ** 0.5
                applied_offset = int(mean_off)
                status = "OK — LSL sync, ~3ms precision" if std_off < 20 else "WARNING: high variance"
                print(f"  LSL offset: mean={mean_off:+.0f}ms, std={std_off:.1f}ms  [{status}]")
                if applied_offset != 0:
                    for r in rows:
                        t = _safe_int(r.get("t_unix_ms"))
                        if t is not None:
                            corrected = t - applied_offset
                            r["t_unix_ms"] = corrected
                            r["t_iso"] = epoch_to_iso(corrected)
                    for r in rows:
                        t = _safe_int(r.get("t_unix_ms"))
                        r["trial"] = assign_trial(t, trial_boundaries)
                    print(f"  Corrected {len(rows)} timestamps by {-applied_offset:+.0f}ms via LSL")
                ordered_rows = [{col: r.get(col, "") for col in OUTPUT_COLUMNS} for r in rows]
                csv_write(ordered_rows, args.out)
                print(f"Wrote {len(ordered_rows)} rows to {args.out}")
                return

    # ── Method 1: TTL / photodiode ──
    if flash_events and args.ttl_column:
        ttl_col = args.ttl_column
        # Use "_ttl_raw" if the column name matches the auto-detected ones from parsers
        ttl_rows = rows
        ttl_flash_ts = extract_ttl_flash_timestamps(ttl_rows, ttl_col)
        if not ttl_flash_ts:
            # Try the auto-captured _ttl_raw field
            ttl_flash_ts = extract_ttl_flash_timestamps(
                [{**r, ttl_col: r.get("_ttl_raw", "")} for r in rows], ttl_col)
        print(f"TTL column '{ttl_col}': found {len(ttl_flash_ts)} rising edge(s)")
        if len(ttl_flash_ts) >= len(flash_events) * 2 // 3:
            # Match TTL events to browser flash timestamps by order
            paired = list(zip(sorted(ttl_flash_ts), sorted(fe["flash_ts"] for fe in flash_events)))
            ttl_offsets = [et - bt for et, bt in paired]
            mean_off = sum(ttl_offsets) / len(ttl_offsets)
            std_off = (sum((x - mean_off) ** 2 for x in ttl_offsets) / len(ttl_offsets)) ** 0.5
            applied_offset = int(mean_off)
            print(f"  TTL alignment: mean offset={mean_off:+.0f}ms, std={std_off:.1f}ms  "
                  f"[{'OK — hardware sync, ~4ms precision' if std_off < 20 else 'WARNING: high jitter'}]")
            if applied_offset != 0:
                for r in rows:
                    t = _safe_int(r.get("t_unix_ms"))
                    if t is not None:
                        corrected = t - applied_offset
                        r["t_unix_ms"] = corrected
                        r["t_iso"] = epoch_to_iso(corrected)
                for r in rows:
                    t = _safe_int(r.get("t_unix_ms"))
                    r["trial"] = assign_trial(t, trial_boundaries)
                print(f"  Corrected {len(rows)} timestamps by {-applied_offset:+.0f}ms via TTL")
            # Skip all software-based methods
            ordered_rows = [{col: r.get(col, "") for col in OUTPUT_COLUMNS} for r in rows]
            csv_write(ordered_rows, args.out)
            print(f"Wrote {len(ordered_rows)} rows to {args.out}")
            return
        else:
            print(f"  WARNING: only {len(ttl_flash_ts)} TTL events found "
                  f"(expected ~{len(flash_events)}). Falling back to pupil method.")

    # ── Method 2: Pattern-matching (cross-machine, large offset) ──
    coarse_offset: "int | None" = None
    if flash_events and args.cross_machine:
        print("Cross-machine mode: running pattern-matching alignment...")
        all_flash_ts = [fe["flash_ts"] for fe in flash_events]
        coarse_offset = find_coarse_offset_by_flash_pattern(rows, all_flash_ts)
        if coarse_offset is None:
            print("  WARNING: pattern matching failed — "
                  "pupil signal may be too noisy or recording duration too short. "
                  "Consider using --apply-offset to set a manual offset.")

    # Phase 2: per-trial fine alignment.
    # If coarse_offset is known, shift each flash timestamp into eye-tracker clock
    # space before searching, then use a wider ±5000ms window to accommodate
    # residual pattern-match error.
    flash_offsets = []
    for fe in flash_events:
        flash_ts = fe["flash_ts"]
        if coarse_offset is not None:
            # Shift the reference into eye-tracker clock space
            search_center = flash_ts + coarse_offset
            detected_ts = detect_flash_in_pupil(rows, search_center,
                                                search_window_ms=5000)
        else:
            detected_ts = detect_flash_in_pupil(rows, flash_ts)

        if detected_ts > 0:
            # Raw offset = eye_tracker_constriction_time - browser_flash_time.
            # Includes physiological reflex latency (~250ms).
            # Subtract latency so the correction aligns the flash event itself.
            raw_offset = detected_ts - flash_ts
            offset = raw_offset - _PUPIL_REFLEX_LATENCY_MS
            flash_offsets.append(offset)
            print(f"  Flash sync ({fe['trial']}): constriction={detected_ts}, "
                  f"raw_offset={raw_offset:+.0f}ms, "
                  f"latency_corrected_offset={offset:+.0f}ms")
        else:
            src = "coarse offset" if coarse_offset is not None else "per-trial window"
            print(f"  Flash sync ({fe['trial']}): not detected via {src} "
                  f"(blink at trial start or bad pupil data; trial excluded from offset)")

    # Consistency check — high std dev means wrong events are being picked.
    if len(flash_offsets) >= 2:
        mean_off = sum(flash_offsets) / len(flash_offsets)
        std_off = (sum((x - mean_off) ** 2 for x in flash_offsets) / len(flash_offsets)) ** 0.5
        status = "OK" if std_off < 50 else "WARNING: high variance — check pupil data quality"
        print(f"  Offset consistency: mean={mean_off:+.0f}ms, std={std_off:.0f}ms  [{status}]")

    # Compute the offset to apply.
    # Manual offset (from Unix ms photograph) is highest priority — it is a direct
    # measurement of SmartEye_PC_clock − Director_browser_clock via Date.now().
    # Flash-based methods are fallback for when no manual measurement is available.
    applied_offset = 0
    if args.apply_offset is not None:
        applied_offset = args.apply_offset
        print(f"Applying manual offset: {applied_offset:+.0f}ms (SmartEye_PC − Director_browser)")
    elif flash_offsets:
        applied_offset = sorted(flash_offsets)[len(flash_offsets) // 2]
        print(f"Applying flash-detected median offset: {applied_offset:+.0f}ms")
    elif coarse_offset is not None:
        applied_offset = coarse_offset - _PUPIL_REFLEX_LATENCY_MS
        print(f"Falling back to coarse offset: {applied_offset:+.0f}ms "
              f"(per-trial fine-tuning failed — alignment uncertainty ±8s)")

    # Apply offset to all timestamps (corrects eye tracker timestamps to frontend clock)
    if applied_offset != 0:
        for r in rows:
            t = _safe_int(r.get("t_unix_ms"))
            if t is not None:
                corrected = t - int(applied_offset)
                r["t_unix_ms"] = corrected
                r["t_iso"] = epoch_to_iso(corrected)
        # Re-assign trials with corrected timestamps
        for r in rows:
            t = _safe_int(r.get("t_unix_ms"))
            r["trial"] = assign_trial(t, trial_boundaries)
        print(f"Corrected {len(rows)} timestamps by {-applied_offset:+.0f}ms")

    ordered_rows = []
    for r in rows:
        ordered_rows.append({col: r.get(col, "") for col in OUTPUT_COLUMNS})

    csv_write(ordered_rows, args.out)
    print(f"Wrote {len(ordered_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
