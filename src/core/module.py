import threading
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from typing import Any, Callable, Dict, Mapping, final

import zmq

from src.core.events import ModuleEvent
from src.tools.logger import logging


class Module(ABC):
    def __init__(self):
        """Child Modules must call super.__init__() in their __init__() function."""
        self.ctx = None
        self.pub_socket = None
        self.connect_hostname = None
        self.xpub_port = None
        self.xsub_port = None
        self.subs: Dict[str, zmq.Socket[bytes]] = {}
        self.callbacks = {}
        self._poller_running = False
        self.poller = None
        self.logger = logging.getLogger(__name__)

    @final
    def _initialize(self) -> None:
        """
        Called inside start_module() or manually before usage.
        This function exist because ctx cannot be set in __init__, because of multi-processing. maybe deprecated
        """
        self.ctx = zmq.Context()
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{self.connect_hostname}:{self.xpub_port}")
        self.poller = threading.Thread(target=self._poll_loop, daemon=True)
        self.set_subscriptions()

    @abstractmethod
    def set_subscriptions(self) -> None:
        """Child module must define this funcction with subscriptions"""
        ...

    @final
    def subscribe(self, topic: str, callback: Callable) -> None:
        sub_socket = self.ctx.socket(zmq.SUB)
        sub_socket.connect(f"tcp://{self.connect_hostname}:{self.xsub_port}")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.subs[topic] = sub_socket
        self.callbacks[topic] = callback
        self.logger.info(f"Subscribe: {topic}")

    @final
    def publish(
        self,
        topic: str,
        **kwargs: Mapping[str, Any],
    ) -> None:
        """
        Will publish a ModuleEvent to other modules.

        :param topic: the topic of the event
        :type topic: str
        :param kwargs: kwargs must be named as the receiving module's callbacks
        :type kwargs: Mapping[str, Any]
        """
        event = ModuleEvent(topic=topic, payload=kwargs)
        self.logger.info(f"Publish: {topic} {kwargs.keys()}")
        self.pub_socket.send_multipart(event.serialize())

    @final
    def _start_polling(self) -> None:
        self._poller_running = True
        self.poller.start()

    @final
    def _poll_loop(self) -> None:
        poller = zmq.Poller()
        for sub in self.subs.values():
            poller.register(sub, zmq.POLLIN)

        while self._poller_running:
            events = dict(poller.poll(100))
            for _, sub in self.subs.items():
                if sub in events:
                    data = sub.recv_multipart()
                    event = ModuleEvent.deserialize(data)

                    self.logger.info(f"Receive: {event.topic} {event.payload.keys()}")
                    self.callbacks[event.topic](**event.payload)

    @final
    def start_module(
        self,
        connect_hostname: str,
        xpub_port: int,
        xsub_port: int,
        stop_event: Event = None,
    ) -> None:
        self.connect_hostname = connect_hostname
        self.xpub_port = xpub_port
        self.xsub_port = xsub_port
        self._initialize()
        if self.subs != {}:
            self._start_polling()
        try:
            self.run_module(stop_event)
        except KeyboardInterrupt:
            self.logger.info("Ctrl+C pressed, exiting cleanly")
        except Exception as e:
            self.logger.error(e)
        finally:
            self.stop_module()

    @final
    def stop_module(self) -> None:
        """Stop the module gracefully."""

        if self._poller_running:
            self._poller_running = False
            self.poller.join()

        for topic, sub in self.subs.items():
            try:
                sub.close(0)
            except Exception as e:
                self.logger.error(f"Error closing SUB socket for '{topic}': {e}")

        self.subs.clear()
        self.callbacks.clear()

        try:
            self.pub_socket.close(0)
        except Exception as e:
            self.logger.error(f"Error closing SUB socket for '{topic}': {e}")

        try:
            self.ctx.term()
        except Exception as e:
            self.logger.error(f"Error terminating ZMQ context: {e}")

        self.logger.info("Module stopped gracefully.")

    def run_module(self, stop_event: Event = None) -> None:
        """Child modules override this instead of run(). Default: idle wait."""
        if stop_event:
            stop_event.wait()

    @final
    def set_custom_logger(self, logger) -> None:
        """The default logger in set in __init__."""
        self.logger = logger
