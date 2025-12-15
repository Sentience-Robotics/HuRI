import json
import threading
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from typing import Callable, Dict, final

import zmq

from src.tools.logger import logging

# from .zmq.event_proxy import "6665", XSUB_ENDPOINT


class Module(ABC):
    def __init__(self):
        """Child Modules must call super.__init__() in their __init__() function."""
        self.ctx = None
        self.pub_socket = None
        self.subs: Dict[str, zmq.Socket[bytes]] = {}
        self.callbacks = {}
        self._poller_running = False
        self.poller = None
        self.logger = logging.getLogger(__name__)

    @final
    def _initialize(self) -> None:
        """
        Called inside start_module() or manually before usage.
        This function exist because ctx cannot be set in __init__, because of multi-processing.
        """
        self.ctx = zmq.Context()
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.connect("tcp://localhost:6666")
        self.poller = threading.Thread(target=self._poll_loop, daemon=True)
        self.set_subscriptions()

    @abstractmethod
    def set_subscriptions(self) -> None:
        """Child module must define this funcction with subscriptions"""
        ...

    @final
    def subscribe(self, topic: str, callback: Callable) -> None:
        sub_socket = self.ctx.socket(zmq.SUB)
        sub_socket.connect("tcp://localhost:6665")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.subs[topic] = sub_socket
        self.callbacks[topic] = callback
        self.logger.info(f"Subscribe: {topic}")

    @final
    def publish(
        self, topic: str, msg: object, content_type: str = "str"
    ) -> None:  # TODO content type enum
        if content_type == "json":
            payload = json.dumps(msg).encode()
        elif content_type == "bytes":
            payload = msg
        elif content_type == "str":
            payload = msg.encode()
        else:
            raise ValueError(f"Unsupported content_type: {content_type}")

        self.pub_socket.send_multipart([topic.encode(), content_type.encode(), payload])
        self.logger.info(f"Publish: {topic} {content_type}")

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
                    topic, content_type, payload = sub.recv_multipart()
                    topic_str = topic.decode()
                    content_type_str = content_type.decode()
                    self.logger.info(f"Receive: {topic_str} {content_type_str}")
                    if content_type_str == "json":
                        kwargs = json.loads(payload.decode())
                        self.callbacks[topic_str](
                            **kwargs
                        )  # TODO better and cleaner way ?
                    elif content_type_str == "bytes":
                        data = payload
                        self.callbacks[topic_str](data)
                    elif content_type_str == "str":
                        data = payload.decode()
                        self.callbacks[topic_str](data)

    @final
    def start_module(self, stop_event: Event = None) -> None:
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

        self.logger.info(f"Module stopped gracefully.")

    def run_module(self, stop_event: Event = None) -> None:
        """Child modules override this instead of run(). Default: idle wait."""
        if stop_event:
            stop_event.wait()

    @final
    def set_custom_logger(self, logger) -> None:
        """The default logger in set in __init__."""
        self.logger = logger
