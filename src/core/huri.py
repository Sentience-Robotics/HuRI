import uuid
from typing import Dict

from fastapi import FastAPI, WebSocket
from ray import serve
from ray.serve import handle

from src.modules.speech_to_text.record_speech import MIC
from src.modules.speech_to_text.speech_to_text import STT
from src.modules.utils.sender import Sender

from .session import Session

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class HuRI:
    def __init__(self, config, handles: Dict[str, handle.DeploymentHandle]) -> None:
        self.config = config
        self.handles = handles

        self.clients: Dict[str, Session] = {}

    @app.websocket("/session")
    async def run_session(self, ws: WebSocket):
        await ws.accept()

        modules = [
            STT(self.handles["stt"]),
            MIC(5),
            Sender(ws, "text"),
        ]
        session_id = str(uuid.uuid4())

        self.clients[session_id] = Session(modules)

        async def receive_loop(session: Session, ws: WebSocket):
            while True:
                msg = await ws.receive()
                if "bytes" in msg:
                    chunk = msg["bytes"]
                    await session.publish("chunk", chunk)
                # else:
                #     data = msg
                #     await session.publish(data["type"], data["data"])

        await receive_loop(self.clients[session_id], ws)
