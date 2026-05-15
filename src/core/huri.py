import uuid
from typing import Dict, List, Type

from fastapi import WebSocket, WebSocketDisconnect
from ray import serve
from ray.serve import handle

from src.modules.factory import Module, ModuleFactory
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
    ) -> None:
        self.factory = ModuleFactory(handles)
        for name, module_cls in modules.items():
            self.factory.register(name, module_cls)
        self.clients: Dict[str, Session] = {}

    @app.websocket("/session")
    async def run_session(self, ws: WebSocket):
        await ws.accept()
        client_config_raw: Dict = await ws.receive_json()
        client_config = ClientConfig.from_dict(client_config_raw)

        _user_id = client_config_raw.get("_user_id") or str(uuid.uuid4())

        senders: List[Module] = [
            Sender(ws, topic) for topic in client_config.topic_list
        ]
        modules: List[Module] = (
            self.factory.create_from_config(_user_id, client_config.modules) + senders
        )

        await ws.send_json({"type": "session_init", "_user_id": _user_id})

        session_id = str(uuid.uuid4())
        self.clients[session_id] = Session(modules)
        print(f"Client registered with _user_id={_user_id}, config: {client_config}")

        async def receive_loop(session: Session, ws: WebSocket):
            try:
                while True:
                    msg = await ws.receive()
                    if "bytes" in msg:
                        chunk = msg["bytes"]
                        await session.publish("chunk", chunk)
                    # else:
                    #     data = msg
                    #     await session.publish(data["type"], data["data"])
            except (WebSocketDisconnect, RuntimeError):
                print(f"Client {_user_id} disconnected")

        await receive_loop(self.clients[session_id], ws)
        del self.clients[session_id]