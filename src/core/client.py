import asyncio
import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Type

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
    ):
        self.config = config
        self.user_id_file = user_id_file
        self.senders_dict = senders_dict

    def _load_user_id(self) -> Optional[str]:
        if os.path.exists(self.user_id_file):
            with open(self.user_id_file) as f:
                return f.read().strip()
        return None

    def _save_user_id(self, _user_id: str):
        with open(self.user_id_file, "w") as f:
            f.write(_user_id)

    async def _receive_loop(self, ws: websockets.ClientConnection):
        import struct
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
                        n_samples = (len(payload) - 13) // 4
                        print(
                            f"<< audio: pts={pts:.3f}s samples={n_samples} @ {sample_rate}Hz "
                            f"end={bool(end_flag)}"
                        )
                    elif topic == "motion" and len(payload) >= 16:
                        pts, fps, n_frames = struct.unpack(">dII", payload[:16])
                        print(f"<< motion: pts={pts:.3f}s frames={n_frames} @ {fps}fps")
                    else:
                        print(f"<< {topic}: bytes ({len(payload)}B)")
                else:
                    print("<<", msg)

        except (asyncio.CancelledError, websockets.ConnectionClosedOK):
            pass

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
