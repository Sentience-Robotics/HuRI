from dataclasses import dataclass
from typing import Optional, Mapping, Any

import zmq

from src.tools.logger import logging, setup_logger
from src.core.events import ModuleEvent


@dataclass
class ZMQEventPorts:
    xpub: str
    xsub: str


class EventProxy:
    def __init__(
        self,
        hostname: str,
        connect_hostname: str,
        xpub_port: int,
        xsub_port: int,
        logger: Optional[logging.Logger] = setup_logger("EventProxy"),
    ):

        self.ctx = zmq.Context.instance()
        self.xpub = self.ctx.socket(zmq.XPUB)
        self.xsub = self.ctx.socket(zmq.XSUB)

        self.hostname = hostname
        self.connect_hostname = connect_hostname
        self.xpub_port = xpub_port
        self.xsub_port = xsub_port

        self.logger = logger or logging.getLogger(__name__)

    def start(self, xpub_connect: bool, xsub_connect: bool):
        if xpub_connect:
            self.xpub.connect(f"tcp://{self.connect_hostname}:{self.xpub_port}")
        else:
            self.xpub.bind(f"tcp://{self.hostname}:{self.xpub_port}")
        if xsub_connect:
            self.xsub.connect(f"tcp://{self.connect_hostname}:{self.xsub_port}")
        else:
            self.xsub.bind(f"tcp://{self.hostname}:{self.xsub_port}")

        try:
            self.logger.info("Correctly initialized, starting proxy")
            zmq.proxy(self.xsub, self.xpub)
        except Exception as e:
            self.logger.error(e)

    def stop(self) -> None:
        self.xsub.close(linger=0)
        self.xpub.close(linger=0)

    def publish(
        self,
        topic: str,
        **kwargs: Mapping[str, Any],
    ) -> None:
        event = ModuleEvent(topic=topic, payload=kwargs)
        self.xpub.send_multipart(event.serialize())
        self.logger.info(f"Publish: {topic} str")
