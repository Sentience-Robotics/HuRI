import asyncio
import json
import os
import struct
import wave
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Type

import numpy as np
import websockets

from src.core.dataclasses.config import ClientConfig

from .client_senders import ClientSender, get_senders


class Client:
    """Client is init with a Config, and connects to HuRI using websockets"""

    def __init__(
        self,
        config: ClientConfig,
        user_id_file: str = os.path.expanduser("~/.huri_user_id"),
        senders_dict: Dict[str, Type[ClientSender]] = get_senders(),
        save_audio_dir: Optional[str] = None,
    ):
        self.config = config
        self.user_id_file = user_id_file
        self.senders_dict = senders_dict

        # When set, incoming audio chunks are buffered per utterance and written
        # to a .wav under this directory each time an end-of-utterance marker
        # arrives — handy for ear-checking what the TTS actually streamed.
        self.save_audio_dir = save_audio_dir
        self._audio_buf: List[np.ndarray] = []
        self._audio_sr: Optional[int] = None
        self._audio_idx = 0
        if save_audio_dir:
            os.makedirs(save_audio_dir, exist_ok=True)

    def _load_user_id(self) -> Optional[str]:
        if os.path.exists(self.user_id_file):
            with open(self.user_id_file) as f:
                return f.read().strip()
        return None

    def _save_user_id(self, _user_id: str):
        with open(self.user_id_file, "w") as f:
            f.write(_user_id)

    def _collect_audio(self, samples: np.ndarray, sample_rate: int, end: bool) -> None:
        if samples.size:
            self._audio_buf.append(samples)
            self._audio_sr = sample_rate
        if end:
            self._flush_audio()

    def _flush_audio(self) -> None:
        if not self._audio_buf or self._audio_sr is None:
            self._audio_buf = []
            return
        audio = np.concatenate(self._audio_buf)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(
            self.save_audio_dir, f"utt-{self._audio_idx:03d}-{stamp}.wav"
        )
        self._write_wav(path, audio, self._audio_sr)
        print(
            f"** saved audio: {path} ({audio.size} samples, "
            f"~{audio.size / self._audio_sr:.2f}s @ {self._audio_sr}Hz)"
        )
        self._audio_idx += 1
        self._audio_buf = []

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
        # float32 [-1, 1] -> 16-bit PCM, clipped to avoid wraparound on overshoot.
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

    async def _receive_loop(self, ws: websockets.ClientConnection):
        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    if len(msg) < 2:
                        print(f"<< bytes ({len(msg)}B, no topic)")
                        continue
                    (topic_len,) = struct.unpack(">H", msg[:2])
                    topic = msg[2:2 + topic_len].decode()
                    payload = msg[2 + topic_len:]

                    if topic == "audio" and len(payload) >= 13:
                        sample_rate, end_flag, pts = struct.unpack(">IBd", payload[:13])
                        # Samples are native-endian float32 (Sender uses ndarray.tobytes()).
                        samples = np.frombuffer(payload[13:], dtype=np.float32)
                        print(
                            f"<< audio: pts={pts:.3f}s samples={samples.size} @ {sample_rate}Hz "
                            f"end={bool(end_flag)}"
                        )
                        if self.save_audio_dir:
                            self._collect_audio(samples, sample_rate, bool(end_flag))
                    elif topic == "motion" and len(payload) >= 16:
                        pts, fps, n_frames = struct.unpack(">dII", payload[:16])
                        print(f"<< motion: pts={pts:.3f}s frames={n_frames} @ {fps}fps")
                    else:
                        print(f"<< {topic}: bytes ({len(payload)}B)")
                else:
                    print("<<", msg)

        except (asyncio.CancelledError, websockets.ConnectionClosedOK):
            pass
        finally:
            if self.save_audio_dir:
                self._flush_audio()  # save anything left if the stream ended mid-utterance

    async def run(self):
        async with websockets.connect(self.config.huri_url) as ws:
            print("Connected to server")

            self.config.user_id = self._load_user_id()

            senders: List[ClientSender] = [
                self.senders_dict[config.name](ws=ws, **config.args)
                for config in self.config.senders.values()
            ]

            await ws.send(json.dumps(asdict(self.config)))

            init_msg = json.loads(await ws.recv())
            if init_msg.get("type") == "session_init":
                user_id = init_msg["user_id"]
                self._save_user_id(user_id)
                print(f"Session started with _user_id: {user_id}")

            receive_task = asyncio.create_task(self._receive_loop(ws))
            await asyncio.gather(
                *(sender.input_loop() for sender in senders),
            )

            receive_task.cancel()
