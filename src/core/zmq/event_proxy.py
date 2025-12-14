from dataclasses import dataclass
from typing import Optional

import zmq

from src.tools.logger import logging, setup_logger


@dataclass
class ZMQEventPorts:
    xpub: str
    xsub: str


class EventProxy:
    def __init__(
        self,
        ports: ZMQEventPorts,
        logger: Optional[logging.Logger] = setup_logger("EventProxy"),
    ):

        self.ctx = zmq.Context.instance()
        self.xpub = self.ctx.socket(zmq.XPUB)
        self.xsub = self.ctx.socket(zmq.XSUB)
        self.ports = ports

        self.logger = logger or logging.getLogger(__name__)

    def start(self, xpub_connect: bool, xsub_connect: bool):
        if xpub_connect:
            self.xpub.connect(f"tcp://localhost:{self.ports.xpub}")
        else:
            self.xpub.bind(f"tcp://localhost:{self.ports.xpub}")
        if xsub_connect:
            self.xsub.connect(f"tcp://localhost:{self.ports.xsub}")
        else:
            self.xsub.bind(f"tcp://localhost:{self.ports.xsub}")

        try:
            self.logger.info("Correctly initialized, starting proxy")
            zmq.proxy(self.xsub, self.xpub)
        except Exception as e:
            self.logger.error(e)

    def stop(self) -> None:
        self.xsub.close(linger=0)
        self.xpub.close(linger=0)
