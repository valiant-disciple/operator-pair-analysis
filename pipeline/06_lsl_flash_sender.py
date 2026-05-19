#!/usr/bin/env python3
"""
LSL Flash Sender — runs on the DIRECTOR'S BROWSER MACHINE.

Listens for flash events from the browser over a local WebSocket, then
re-emits them as an LSL marker stream discoverable on the LAN.

The Director's eye tracker machine runs lsl_flash_receiver.py to pick up
these markers with timestamps already corrected to the eye tracker machine's
local clock (LSL handles the cross-machine clock correction automatically).

Usage:
  pip install pylsl websockets
  python lsl_flash_sender.py [--port 9001]

Then open the Director page in the browser — it will connect automatically.
Keep this running for the entire session.
"""

import argparse
import asyncio
import json
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("Missing dependency: pip install websockets")

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError:
    sys.exit("Missing dependency: pip install pylsl")


def make_outlet() -> StreamOutlet:
    info = StreamInfo(
        name="FlashMarkers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,          # irregular rate
        channel_format="string",
        source_id="map_task_flash_sender",
    )
    return StreamOutlet(info)


async def handle_client(websocket, outlet: StreamOutlet):
    client = websocket.remote_address
    print(f"  Browser connected: {client}")
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") != "flash":
                continue

            trial = msg.get("trialIndex", "?")
            flash_ts = msg.get("flashTs", 0)
            session_id = msg.get("sessionId", "")

            # Encode all info into the marker string so the receiver can log it.
            marker = f"flash|{trial}|{flash_ts}|{session_id}"
            outlet.push_sample([marker])

            lsl_t = local_clock()
            unix_ms = int(time.time() * 1000)
            print(f"  Marker sent: trial={trial}  browser_ts={flash_ts}  "
                  f"local_unix={unix_ms}  lsl_clock={lsl_t:.4f}")

    except websockets.ConnectionClosed:
        pass
    finally:
        print(f"  Browser disconnected: {client}")


async def main(port: int):
    outlet = make_outlet()
    print(f"LSL stream 'FlashMarkers' created — discoverable on LAN")
    print(f"Waiting for browser on ws://localhost:{port} ...")

    async with websockets.serve(
        lambda ws: handle_client(ws, outlet),
        "localhost",
        port,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LSL flash marker sender")
    ap.add_argument("--port", type=int, default=9001,
                    help="Local WebSocket port the browser connects to (default: 9001)")
    args = ap.parse_args()
    try:
        asyncio.run(main(args.port))
    except KeyboardInterrupt:
        print("\nStopped.")
