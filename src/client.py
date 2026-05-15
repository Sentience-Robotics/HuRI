import argparse
import asyncio
import json
import os
from dataclasses import asdict
from typing import Dict

import numpy as np
import sounddevice as sd
import websockets
from omegaconf import OmegaConf

from src.core.dataclasses.config import ClientConfig


USER_ID_FILE = os.path.expanduser("~/.huri_user_id")


def load_user_id() -> str | None:
    if os.path.exists(USER_ID_FILE):
        with open(USER_ID_FILE) as f:
            return f.read().strip()
    return None


def save_user_id(_user_id: str):
    with open(USER_ID_FILE, "w") as f:
        f.write(_user_id)

def load_client_config(path: str) -> ClientConfig:
    with open(path) as f:
        dict_config = OmegaConf.load(f)
    raw_resolved = OmegaConf.to_container(dict_config, resolve=True)

    if not isinstance(raw_resolved, Dict):
        raise RuntimeError("error yaml does not output a dict")

    return ClientConfig.from_dict(raw_resolved)


async def stream_audio():
    parser = argparse.ArgumentParser(description="Client config")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to Client config file (YAML)",
    )

    args = parser.parse_args()
    config = load_client_config(args.config)

    FRAME_SIZE = int(config.sample_rate * config.frame_duration)
    async with websockets.connect(config.huri_url) as ws:
        print("Connected to server")

        payload = asdict(config)
        _user_id = load_user_id()
        if _user_id:
            payload["_user_id"] = _user_id
            print(f"Reconnecting with _user_id: {_user_id}")

        await ws.send(json.dumps(payload))

        init_msg = json.loads(await ws.recv())
        if init_msg.get("type") == "session_init":
            _user_id = init_msg["_user_id"]
            save_user_id(_user_id)
            print(f"Session started with _user_id: {_user_id}")

        async def receive(ws: websockets.ClientConnection):
            while True:
                text = await ws.recv()
                print("received:", text)

        async def send(ws: websockets.ClientConnection):
            loop = asyncio.get_running_loop()

            queue: asyncio.Queue = asyncio.Queue()

            def callback(indata: np.ndarray, frames, time, status):
                loop.call_soon_threadsafe(queue.put_nowait, indata.copy())

            with sd.InputStream(
                samplerate=config.sample_rate,
                channels=1,
                dtype="int16",
                callback=callback,
                blocksize=FRAME_SIZE,
            ):
                while True:
                    chunk = await queue.get()
                    await ws.send(chunk.tobytes())

        await asyncio.gather(receive(ws), send(ws))


if __name__ == "__main__":
    asyncio.run(stream_audio())
