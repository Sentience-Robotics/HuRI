# import sounddevice as sd
# import websocket
# import numpy as np
# import time

# # import microphone
# # import stt
# import ray
# from abc import ABC, abstractmethod
# from huri import HuRI
# from ray.actor import ActorHandle
# from fastapi import WebSocket


# class Client(ABC):
#     def __init__(self):

#         # huri = ray.get_actor(name="HuRI", namespace="h")

#         # c = ray.get(huri.new_client.remote())
#         # self.session = ray.get_actor(name=c, namespace="h")

#     @abstractmethod
#     def input(self):
#         pass

#     # @abstractmethod
#     # def output(self):
#     #     pass

#     def ingestion(self, type):
#         self.session.push_in.remote(self.input(), type)

#     def outgestion(self):
#         return ray.get(self.session.get_response.remote())


# class AudioClient(Client):
#     def __init__(self):
#         super().__init__()
#         self.CHUNK_DURATION: float = 0.3
#         self.SAMPLE_RATE: int = 16000

#     def input(self):
#         chunk: np.ndarray = sd.rec(
#             int(self.CHUNK_DURATION * self.SAMPLE_RATE),
#             samplerate=self.SAMPLE_RATE,
#             channels=1,
#             dtype="int16",
#         ).ravel()
#         sd.wait()
#         return chunk


# class TerminalClient(Client):
#     def __init__(self):
#         super().__init__()
#         self.CHUNK_DURATION: float = 0.3
#         self.SAMPLE_RATE: int = 16000

#     def input(self):
#         text = input()
#         return text


# def main():
#     ray.init()

#     client = TerminalClient()
#     while True:
#         client.ingestion()
#         print(client.outgestion())


import asyncio
import time
import wave

import fastapi
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

            queue = asyncio.Queue()

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
