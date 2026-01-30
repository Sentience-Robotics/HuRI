import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import zmq

from src.tools.logger import logging, setup_logger


@dataclass
class Command:
    cmd: str  # "STOP", "START", "STATUS", ...
    args: List[Any]  # JSON-serializable arguments

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> "Command":
        obj = json.loads(data.decode("utf-8"))
        return Command(**obj)


@dataclass
class Result:
    success: bool
    result: List[Any]

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> "Command":
        obj = json.loads(data.decode("utf-8"))
        return Result(**obj)


class Router:
    def __init__(
        self,
        hostname: str,
        port: int,
        logger: Optional[logging.Logger] = setup_logger("Router"),
    ):

        self.ctx = zmq.Context.instance()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.hostname = hostname
        self.port = port

        self.logger = logger or logging.getLogger(__name__)

        self.dealers: Dict[bytes, bool] = {}

    def start(self):
        self.router.bind(f"tcp://{self.hostname}:{self.port}")
        self.logger.info("Router started")

        try:
            while True:
                identity, *frames = self.router.recv_multipart()

                if not frames:
                    continue

                command = frames[0]

                if command == b"REGISTER":
                    self.dealers[identity] = True
                    self.logger.info(f"Dealer registered: {identity}")

                elif command == b"RESULT":
                    payload = frames[1] if len(frames) > 1 else b""
                    self.logger.info(f"Result from {identity}: {payload.decode()}")
        except Exception as e:
            self.logger.exception(e)
            pass
        finally:
            self.router.close()

    def stop(self) -> None:
        self.router.close()

    def send_command(self, dealer_id: bytes, command: Command) -> None:
        if dealer_id not in self.dealers:
            raise ValueError("Dealer not registered")

        self.router.send_multipart([dealer_id, b"COMMAND", command.to_bytes()])

    def send_commands(self, command: Command) -> None:
        for dealer_id, _ in self.dealers.items():
            self.send_command(dealer_id, command)


class Dealer:
    def __init__(
        self,
        hostname: str,
        port: int,
        executor: Callable[[Command], bool],
        logger: Optional[logging.Logger] = None,
        identity: Optional[str] = None,
    ):
        self.ctx = zmq.Context.instance()
        self.dealer = self.ctx.socket(zmq.DEALER)

        self.hostname = hostname
        self.port = port

        self.executor = executor
        self.identity = (identity or str(uuid.uuid4())).encode()  # TODO agent name

        self.logger = logger or logging.getLogger(f"Dealer {self.identity}")

    def start(self):
        self.dealer.connect(f"tcp://{self.hostname}:{self.port}")
        self.dealer.setsockopt(zmq.IDENTITY, self.identity)
        self.logger.info(f"Dealer started: {self.identity}")

        try:
            self.dealer.send(b"REGISTER")

            while True:
                frames = self.dealer.recv_multipart()

                command = frames[0]

                if command == b"COMMAND":
                    self.logger.info("received command")
                    payload = frames[1] if len(frames) > 1 else b""
                    result = self.execute(payload)

                    self.dealer.send_multipart([b"RESULT", result])
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.dealer.close()

    def execute(self, command: Command) -> bytes:
        """
        Execute command sent by Router
        """
        self.executor(command)

        # Example execution
        result = f"Executed: {command.cmd}"
        return result.encode()

    def stop(self) -> None:
        self.dealer.close(linger=0)
