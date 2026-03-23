import argparse
import asyncio
import json
from dataclasses import asdict

import numpy as np
import sounddevice as sd
import websockets
import yaml

from src.core.dataclasses.config import ClientConfig


def load_client_config(path: str) -> ClientConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

        return ClientConfig.from_dict(raw)


CHUNK_DURATION = 1
SAMPLE_RATE = 16000


async def stream_audio():
    parser = argparse.ArgumentParser(description="Client config")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to Client config file (YAML)",
    )

    args = parser.parse_args()
    config = load_client_config(args.config)

    async with websockets.connect(config.huri_url) as ws:
        print("Connected to server")

        await ws.send(json.dumps(asdict(config)))

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
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                callback=callback,
                blocksize=int(CHUNK_DURATION * SAMPLE_RATE),
            ):
                while True:
                    chunk = await queue.get()
                    await ws.send(chunk.tobytes())

        await asyncio.gather(receive(ws), send(ws))


if __name__ == "__main__":
    asyncio.run(stream_audio())
