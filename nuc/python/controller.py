# controller_server_once.py
# A tiny one-shot WebSocket server:
# - waits for exactly one message from Unity
# - prints it
# - replies with a small ACK
# - exits

import asyncio
import json
import sys
from datetime import datetime

import websockets


HOST = "127.0.0.1"
PORT = 7361


async def handler(ws):
    try:
        msg = await ws.recv()
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{ts}] Received from Unity:")
        print(msg)

        # Optional: try JSON parse to validate shape (still prints raw above)
        try:
            parsed = json.loads(msg)
            print("\nParsed JSON:")
            print(json.dumps(parsed, indent=2))
        except Exception:
            print("\n(note: message was not valid JSON)")

        await ws.send(json.dumps({"ok": True, "message": "ACK from Python"}))

    finally:
        # Close server by stopping event loop after handling first message
        asyncio.get_running_loop().call_soon_threadsafe(asyncio.get_running_loop().stop)


async def main():
    print(f"WebSocket listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        # Run until handler stops the loop
        await asyncio.Future()


if __name__ == "__main__":
    # You prefer `python` alias already; run with: python controller_server_once.py
    try:
        asyncio.run(main())
    except RuntimeError:
        # If loop stopped as intended
        pass
    print("Server exiting.")
    sys.exit(0)
