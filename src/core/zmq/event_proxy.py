import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import zmq

from src.core.events import ModuleEvent
from src.tools.logger import logging, setup_logger


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
        self.xpub: zmq.Socket[bytes] = self.ctx.socket(zmq.XPUB)
        self.xsub: zmq.Socket[bytes] = self.ctx.socket(zmq.XSUB)

        self.hostname = hostname
        self.connect_hostname = connect_hostname
        self.xpub_port = xpub_port
        self.xsub_port = xsub_port

        self._started: bool = False

        self.logger = logger or logging.getLogger(__name__)

    def start(self, xpub_connect: bool, xsub_connect: bool):
        """
        Connect to endpoint.
        Launch a proxy thread.
        """
        if self._started is True:
            raise Exception("already started")

        if xpub_connect:
            self.xpub.connect(f"tcp://{self.connect_hostname}:{self.xpub_port}")
        else:
            self.xpub.bind(f"tcp://{self.hostname}:{self.xpub_port}")
        if xsub_connect:
            self.xsub.connect(f"tcp://{self.connect_hostname}:{self.xsub_port}")
        else:
            self.xsub.bind(f"tcp://{self.hostname}:{self.xsub_port}")

        self.logger.info("Correctly initialized, starting proxy")

        self._proxy_thread = threading.Thread(target=self._proxy)
        self._proxy_thread.start()

        self._started = True

    def _proxy(self) -> None:
        try:
            zmq.proxy(self.xsub, self.xpub)  # todo capture to stop
        except Exception as e:
            self.logger.error(e)

    def stop(self) -> None:
        if self._started is False:
            raise Exception("not started")

        self.xsub.close(linger=0)  # todo capture
        self.xpub.close(linger=0)

        self._proxy_thread.join(2.0)
        self._proxy_thread = None

        self._started = False

    def publish(
        self,
        topic: str,
        **kwargs: Mapping[str, Any],
    ) -> None:
        if self._started is False:
            raise Exception("not started")

        event = ModuleEvent(topic=topic, payload=kwargs)
        self.xpub.send_multipart(event.serialize())
        self.logger.info(f"Publish: {topic} str")
