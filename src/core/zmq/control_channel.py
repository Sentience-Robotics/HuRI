import json
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import zmq

from src.core.events import Control, ControlEvent
from src.tools.logger import logging, setup_logger


class Router:
    def __init__(
        self,
        hostname: str,
        port: int,
        handler: Callable[[bytes, ControlEvent], bool],
        logger: Optional[logging.Logger] = setup_logger("Router"),
    ):
        self.ctx = zmq.Context.instance()
        self.router: zmq.Socket[bytes] = self.ctx.socket(zmq.ROUTER)
        self.hostname = hostname
        self.port = port

        self.handler = handler

        self._stop_event = None
        self._poll_thread = None
        self._started = False

        self.dealers: Dict[bytes, bool] = {}

        self.logger = logger or logging.getLogger(__name__)

    def register_dealer(
        self, identity: bytes, auth: str, name: str, config: Dict[str, Any]
    ) -> None:
        if auth != "oui":
            return

        self.dealers[identity] = config
        self.logger.info(f"Dealer registered: {identity}")

    def _poll(self) -> None:  # todo poller
        """
        Start a blocking poller loop.
        call self.stop() to stop.
        """

        while not self._stop_event.is_set():
            try:
                identity, *data = self.router.recv_multipart()
                event = ControlEvent.deserialize(data)

                if self.handler(identity, event) is False:
                    self.logger.warning("Could not execute control")
                else:
                    self.logger.info("Control executed")

            except zmq.Again:
                continue
            except Exception as e:
                self.logger.warning(e, exc_info=True)

    def start(self) -> None:
        """
        Bind to endpoint.
        Launch a poll loop thread.
        """
        if self._started is True:
            raise Exception("already started")

        self.router.bind(f"tcp://{self.hostname}:{self.port}")
        self.router.setsockopt(zmq.RCVTIMEO, 1000)
        self.logger.info("started")

        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll)
        self._poll_thread.start()

        self._started = True

    def stop(self) -> None:
        if self._started is False:
            raise Exception("not started")

        self._stop_event.set()
        self._poll_thread.join(2.0)

        self.router.close(linger=0)
        self._stop_event = None
        self._poll_thread = None

        self._started = False

    def send_control(
        self, dealer_identity: str, Control: Control, **kwargs: Mapping[str, Any]
    ) -> None:
        if self._started is False:
            raise Exception("not started")

        if dealer_identity not in self.dealers:
            raise ValueError(f"Dealer {dealer_identity} not registered")

        event = ControlEvent(cmd=Control, payload=kwargs)
        self.logger.info(f"Sending Control {Control} to: {dealer_identity}")
        self.router.send_multipart([dealer_identity] + event.serialize())

    def send_controls(self, Control: Control, **kwargs: Mapping[str, Any]) -> None:
        if self._started is False:
            raise Exception("not started")

        for dealer_identity, _ in self.dealers.items():
            self.send_control(dealer_identity, Control, **kwargs)


class Dealer:  # todo heartbeat
    def __init__(
        self,
        hostname: str,
        port: int,
        handler: Callable[[Control], bool],
        logger: Optional[logging.Logger] = None,
        identity: Optional[str] = None,
    ):
        self.ctx = zmq.Context.instance()
        self.dealer: zmq.Socket[bytes] = self.ctx.socket(zmq.DEALER)

        self.hostname = hostname
        self.port = port

        self.handler = handler
        self.identity = identity or str(uuid.uuid4())  # TODO agent name

        self._stop_event = None
        self._poll_thread = None
        self._started = False

        self.logger = logger or logging.getLogger(f"Dealer {self.identity}")

    def _poll(self) -> None:
        """
        Start a blocking poller loop.
        call self.stop() to stop.
        """

        while not self._stop_event.is_set():
            try:
                data = self.dealer.recv_multipart()
                Control = ControlEvent.deserialize(data)

                self.logger.info(f"Received Control {Control.cmd}")
                result = self.handler(Control)

                # self.dealer.send_multipart([b"RESULT", result])
            except zmq.Again:
                continue
            except Exception as e:
                self.logger.exception(e)

    def start(self) -> None:
        """
        Connect to endpoint.
        Launch a poll loop thread.
        Send Register Control (wip).
        """
        if self._started is True:
            raise Exception("already started")

        self.dealer.connect(f"tcp://{self.hostname}:{self.port}")
        self.dealer.setsockopt(zmq.IDENTITY, b"name")
        self.dealer.setsockopt(zmq.RCVTIMEO, 1000)
        self.logger.info(f"Dealer started: {self.identity}")

        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll)
        self._poll_thread.start()

        register = ControlEvent(
            cmd=Control.REGISTER,
            payload={
                "auth": "oui",
                "name": self.identity,
                "config": {"none": None},
            },
        )
        self.dealer.send_multipart(register.serialize())

        self._started = True

    def stop(self) -> None:
        if self._started is False:
            raise Exception("not started")

        self._stop_event.set()
        self._poll_thread.join(2.0)

        self.dealer.close(linger=0)
        self._stop_event = None
        self._poll_thread = None

        self._started = False
