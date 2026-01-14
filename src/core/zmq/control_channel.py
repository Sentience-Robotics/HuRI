import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Mapping

import zmq

from src.core.events import Command, CommandEvent
from src.tools.logger import logging, setup_logger

# @dataclass
# class Command:
#     cmd: str  # "STOP", "START", "STATUS", ...
#     args: List[Any]  # JSON-serializable arguments

#     def to_bytes(self) -> bytes:
#         return json.dumps(asdict(self)).encode("utf-8")

#     @staticmethod
#     def from_bytes(data: bytes) -> "Command":
#         obj = json.loads(data.decode("utf-8"))
#         return Command(**obj)


# @dataclass
# class Result:
#     success: bool
#     result: List[Any]

#     def to_bytes(self) -> bytes:
#         return json.dumps(asdict(self)).encode("utf-8")

#     @staticmethod
#     def from_bytes(data: bytes) -> "Command":
#         obj = json.loads(data.decode("utf-8"))
#         return Result(**obj)


class Router:
    def __init__(
        self,
        hostname: str,
        port: int,
        logger: Optional[logging.Logger] = setup_logger("Router"),
    ):
        self.ctx = zmq.Context.instance()
        self.router: zmq.Socket[bytes] = self.ctx.socket(zmq.ROUTER)
        self.hostname = hostname
        self.port = port

        self.logger = logger or logging.getLogger(__name__)

        self.dealers: Dict[bytes, bool] = {}

    def register_dealer(self, auth: str, name: str, config: Dict[str, Any]) -> None:
        if auth != "oui":
            return

        self.dealers[name] = config
        self.logger.info(f"Dealer registered: {name}")

    def start(self):
        self.router.bind(f"tcp://{self.hostname}:{self.port}")
        self.logger.info("Router started")

        try:
            while True:
                identity, *data = self.router.recv_multipart()
                self.logger.warning(data)
                command = CommandEvent.deserialize(data)

                if command.cmd == Command.REGISTER:
                    self.register_dealer(**command.payload)

        except Exception as e:
            self.logger.exception(e)
            pass
        finally:
            self.router.close()

    def stop(self) -> None:
        self.router.close()

    def send_command(
        self, dealer_name: str, command: Command, **kwargs: Mapping[str, Any]
    ) -> None:
        if dealer_name not in self.dealers:
            raise ValueError("Dealer not registered")

        event = CommandEvent(cmd=command, payload=kwargs)
        self.router.send_multipart(event.serialize())

    def send_commands(self, command: Command, **kwargs: Mapping[str, Any]) -> None:
        for dealer_name, _ in self.dealers.items():
            self.logger.info(f"Sending Command to: {dealer_name}")
            self.send_command(dealer_name, command, **kwargs)


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
        self.dealer: zmq.Socket[bytes] = self.ctx.socket(zmq.DEALER)

        self.hostname = hostname
        self.port = port

        self.executor = executor
        self.identity = identity or str(uuid.uuid4())  # TODO agent name

        self.logger = logger or logging.getLogger(f"Dealer {self.identity}")

    def start(self):
        self.dealer.connect(f"tcp://{self.hostname}:{self.port}")
        self.dealer.setsockopt(zmq.IDENTITY, self.identity.encode())
        self.logger.info(f"Dealer started: {self.identity}")

        try:
            register = CommandEvent(
                cmd=Command.REGISTER,
                payload={
                    "auth": "oui",
                    "name": self.identity,
                    "config": {"none": None},
                },
            )
            self.dealer.send_multipart(register.serialize())

            while True:
                # self.dealer.
                self.logger.info("received nothing still")
                data = self.dealer.recv_multipart()
                self.logger.info("received")
                command = CommandEvent.deserialize(data)

                self.logger.info("received command")
                result = self.execute(command)

                # self.dealer.send_multipart([b"RESULT", result])
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.dealer.close()

    def execute(self, command: CommandEvent) -> bytes:
        """
        Execute command sent by Router
        """
        self.executor(command)

        # Example execution
        result = f"Executed: {command.cmd}"
        return result.encode()

    def stop(self) -> None:
        self.dealer.close(linger=0)
