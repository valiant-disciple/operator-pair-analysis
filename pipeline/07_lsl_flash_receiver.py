#!/usr/bin/env python3
"""
LSL Flash Receiver — runs on the DIRECTOR'S EYE TRACKER MACHINE.

Resolves the 'FlashMarkers' LSL stream from the browser machine and records
each flash event with a timestamp that is automatically corrected to THIS
machine's local clock by LSL's cross-machine clock synchronisation.

The resulting flash_markers.csv is passed to preprocess_eye.py via
--lsl-markers-file to align the eye tracker data to the experiment timeline
with ~3ms precision.

Usage:
  pip install pylsl
  python lsl_flash_receiver.py --out flash_markers.csv [--session SESSION_ID]

Keep this running before the session starts and stop it after the last trial.
Both machines must be on the same LAN (same subnet, LSL multicast UDP).
"""

import argparse
import csv
import sys
import time

try:
    from pylsl import StreamInlet, resolve_stream, local_clock
except ImportError:
    sys.exit("Missing dependency: pip install pylsl")


def lsl_ts_to_unix_ms(lsl_ts: float) -> int:
    """
    Convert an LSL timestamp (local_clock domain) to Unix milliseconds on
    THIS machine. LSL timestamps are already corrected for cross-machine
    clock skew by the time they arrive via inlet.pull_sample().
    """
    # offset between LSL epoch and Unix epoch on this machine
    lsl_to_unix = time.time() - local_clock()
    return int((lsl_ts + lsl_to_unix) * 1000)


def main(out_path: str, session_filter: str | None, timeout_s: float):
    print("Resolving LSL stream 'FlashMarkers' on LAN ...")
    print("(Make sure lsl_flash_sender.py is running on the browser machine)")

    streams = resolve_stream("name", "FlashMarkers", timeout=timeout_s)
    if not streams:
        sys.exit(f"ERROR: No 'FlashMarkers' stream found within {timeout_s}s. "
                 "Check that both machines are on the same LAN and the sender is running.")

    inlet = StreamInlet(streams[0])
    print(f"Connected to stream from: {streams[0].hostname()}")
    print(f"Writing markers to: {out_path}")
    print("Waiting for flash events (Ctrl+C to stop)...\n")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "browser_flash_ts_ms", "eye_tracker_unix_ms",
                         "lsl_ts_seconds", "session_id"])

        try:
            while True:
                # pull_sample blocks until a sample arrives.
                # The returned timestamp is already corrected to this machine's
                # local_clock() domain by LSL's cross-machine sync.
                sample, lsl_ts = inlet.pull_sample(timeout=1.0)
                if sample is None:
                    continue

                marker = sample[0]  # e.g. "flash|1|1712345678901|session123"
                parts = marker.split("|")
                trial = parts[1] if len(parts) > 1 else "?"
                browser_ts = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                session_id = parts[3] if len(parts) > 3 else ""

                if session_filter and session_id and session_id != session_filter:
                    continue  # different session, ignore

                eye_tracker_unix_ms = lsl_ts_to_unix_ms(lsl_ts)

                writer.writerow([trial, browser_ts, eye_tracker_unix_ms,
                                 f"{lsl_ts:.6f}", session_id])
                f.flush()

                latency_ms = eye_tracker_unix_ms - browser_ts
                print(f"  Flash: trial={trial}  browser_ts={browser_ts}  "
                      f"eye_tracker_ts={eye_tracker_unix_ms}  "
                      f"apparent_latency={latency_ms:+d}ms")

        except KeyboardInterrupt:
            print("\nStopped. Flash markers saved.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LSL flash marker receiver")
    ap.add_argument("--out", default="flash_markers.csv",
                    help="Output CSV path (default: flash_markers.csv)")
    ap.add_argument("--session", default=None,
                    help="Only record markers for this session ID (optional filter)")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Seconds to wait for the LSL stream before giving up (default: 30)")
    args = ap.parse_args()
    main(args.out, args.session, args.timeout)
