import json
import struct
import uuid
from typing import Dict, List, Type

from fastapi import WebSocket, WebSocketDisconnect
from ray import serve
from ray.serve import handle

from src.modules.factory import EventData, EventDataFactory, Module, ModuleFactory
from src.modules.utils.sender import Sender

from .app import app
from .dataclasses.config import ClientConfig
from .session import Session


@serve.deployment
@serve.ingress(app)
class HuRI:
    def __init__(
        self,
        modules: Dict[str, Type[Module]],
        handles: Dict[str, handle.DeploymentHandle],
        events: Dict[str, Type[EventData]],
    ) -> None:
        self.module_factory = ModuleFactory(handles)
        self.event_factory = EventDataFactory()
        for name, module_cls in modules.items():
            self.module_factory.register(name, module_cls)

            event_cls = events.pop(module_cls.input_type, None)
            self.event_factory.register(module_cls.input_type, event_cls)
            if module_cls.output_type is None:
                continue
            event_cls = events.pop(module_cls.output_type, None)
            self.event_factory.register(module_cls.output_type, event_cls)

        self.clients: Dict[str, Session] = {}

    @app.websocket("/session")
    async def run_session(self, ws: WebSocket):
        await ws.accept()

        client_config_raw: Dict = await ws.receive_json()

        client_config = ClientConfig.from_dict(client_config_raw)

        senders: List[Module] = [
            Sender(ws, topic) for topic in client_config.topic_list
        ]
        modules: List[Module] = (
            self.module_factory.create_from_config(client_config.modules) + senders
        )

        session_id = str(uuid.uuid4())

        self.clients[session_id] = Session(modules)

        print("Client registered successfully with config:", client_config)

        async def receive_loop(session: Session, ws: WebSocket):
            try:
                while True:
                    msg = await ws.receive()

                    if msg["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()

                    if "bytes" in msg:
                        msg_bytes = msg["bytes"]
                        topic_len = struct.unpack("!H", msg_bytes[:2])[0]

                        topic = msg_bytes[2 : 2 + topic_len].decode()
                        data = msg_bytes[2 + topic_len :]
                    else:
                        msg_text = msg["text"]
                        event = json.loads(msg_text)
                        topic = event["topic"]
                        data = event["data"]

                    data = self.event_factory.create(topic, data)

                    await session.publish(topic, data)

            except (WebSocketDisconnect, RuntimeError):
                print(f"Client disconnected: {session_id}")

        await receive_loop(self.clients[session_id], ws)
