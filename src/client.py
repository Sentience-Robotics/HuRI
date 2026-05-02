import asyncio

import numpy as np
import sounddevice as sd
import websockets

SERVER_URL = "ws://localhost:8000/session"
CHUNK_DURATION = 1
SAMPLE_RATE = 16000


async def stream_audio():

    async with websockets.connect(SERVER_URL) as ws:
        print("Connected to server")

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
