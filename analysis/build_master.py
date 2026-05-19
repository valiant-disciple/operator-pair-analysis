#!/usr/bin/env python3
"""
Build master_ml.csv from per-session ml_ready.csv files.

For each session:
- Read ml_ready.csv
- Add session_id (folder name), dyad_id (cleaned), trial_position columns
- Concatenate

Apply trial-level exclusion: stroke count < 3.
Save to: analysis/batch_out/master_ml.csv
Save exclusion log: analysis/batch_out/exclusions.csv
"""
import csv
import json
import os
from pathlib import Path
import re

DSA_ROOT = Path("/Users/kolosus/Documents/DSA")
BATCH_OUT = DSA_ROOT / "analysis" / "batch_out"
MASTER_CSV = BATCH_OUT / "master_ml.csv"
EXCLUSIONS_CSV = BATCH_OUT / "exclusions.csv"


def normalize_dyad_id(folder_name: str) -> str:
    """Convert folder name to consistent dyad ID."""
    # Replace spaces with underscores, lowercase
    return folder_name.lower().strip().replace(' ', '_').replace('-', '_')


def main():
    # Find all per-session ml_ready.csv files
    session_dirs = sorted([
        d for d in BATCH_OUT.iterdir()
        if d.is_dir() and (d / 'ml_ready.csv').exists()
    ])
    print(f"Found {len(session_dirs)} sessions with ml_ready.csv")

    all_rows = []
    excluded = []

    for sess_dir in session_dirs:
        ml_path = sess_dir / 'ml_ready.csv'
        dyad_id = normalize_dyad_id(sess_dir.name)

        with open(ml_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            row['dyad_id'] = dyad_id
            row['session_dir'] = sess_dir.name

            # Try to parse trial position. The 'trial' column may be int or 'TXX' string
            t = row.get('trial', '')
            try:
                trial_int = int(t)
            except (ValueError, TypeError):
                # Skip non-numeric trial labels (e.g., 'T02')
                continue

            row['trial_position'] = trial_int

            # Restrict to data trials (3-8)
            if trial_int < 3 or trial_int > 8:
                continue

            # Apply trial-level exclusion: stroke count < 3
            try:
                strokes = int(row.get('strokes', 0) or 0)
            except (ValueError, TypeError):
                strokes = 0

            if strokes < 3:
                excluded.append({
                    'dyad_id': dyad_id,
                    'session_dir': sess_dir.name,
                    'trial': trial_int,
                    'reason': f'strokes={strokes} < 3',
                    'mapNumber': row.get('mapNumber', ''),
                })
                continue

            all_rows.append(row)

    if not all_rows:
        print("No rows to write")
        return

    # Build full header (union of all keys)
    all_keys = []
    seen_keys = set()
    # Put metadata columns first
    priority = ['dyad_id', 'session_dir', 'trial', 'trial_position', 'sessionId', 'mapNumber']
    for k in priority:
        if any(k in r for r in all_rows):
            all_keys.append(k)
            seen_keys.add(k)
    for r in all_rows:
        for k in r.keys():
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    # Write master CSV
    with open(MASTER_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, '') for k in all_keys})

    print(f"Wrote {MASTER_CSV} with {len(all_rows)} rows × {len(all_keys)} cols")

    # Write exclusions log
    if excluded:
        with open(EXCLUSIONS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dyad_id', 'session_dir', 'trial', 'mapNumber', 'reason'])
            writer.writeheader()
            writer.writerows(excluded)
        print(f"Wrote {EXCLUSIONS_CSV} with {len(excluded)} excluded trials")

    # Summary
    n_dyads = len(set(r['dyad_id'] for r in all_rows))
    print(f"\nFinal: {len(all_rows)} trials × {n_dyads} dyads")
    print(f"Mean trials/dyad: {len(all_rows) / n_dyads:.2f}")


if __name__ == '__main__':
    main()
