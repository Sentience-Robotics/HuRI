import threading
import time
from multiprocessing.synchronize import Event
from typing import Callable, Dict

import zmq

from .event_router import XPUB_ENDPOINT, XSUB_ENDPOINT


class Module:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ctx = zmq.Context()
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.connect(XSUB_ENDPOINT)

        self.subs: Dict[str, zmq.SyncSocket] = {}
        self.callbacks: Dict[str, Callable] = {}
        self._running = False
        self.poller = threading.Thread(target=self._poll_loop, daemon=True)

    def subscribe(self, topic: str, callback: Callable) -> None:
        sub_socket = self.ctx.socket(zmq.SUB)
        sub_socket.connect(XPUB_ENDPOINT)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.subs[topic] = sub_socket
        self.callbacks[topic] = callback

    def publish(self, topic: str, msg: str) -> None:
        self.pub_socket.send_multipart([topic.encode(), msg.encode()])

    def start_polling(self) -> None:
        self._running = True
        self.poller.start()

    def _poll_loop(self) -> None:
        poller = zmq.Poller()
        for sub in self.subs.values():
            poller.register(sub, zmq.POLLIN)

        while self._running:
            events = dict(poller.poll(100))
            for _, sub in self.subs.items():
                if sub in events:
                    topic, msg = sub.recv_multipart()
                    self.callbacks[topic.decode()](msg.decode())

    def run(self, stop_event: Event = None) -> None:
        """
        Default run: waits for events.
        Child classes can override `loop()` for active behavior.
        """
        if self.subs != {}:
            self.start_polling()
        try:
            self.loop(stop_event)
        finally:
            self.stop()

    def loop(self, stop_event: Event = None) -> None:
        """Child modules override this instead of run(). Default: idle wait."""
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.1)

    def stop(self) -> None:
        """Stop the module gracefully."""

        # Close poller daemon
        if self._running:
            self._running = False
            self.poller.join()

        for topic, sub in self.subs.items():
            try:
                sub.close(0)
            except Exception as e:
                print(f"[{self.name}] Error closing SUB socket for '{topic}': {e}")

        self.subs.clear()
        self.callbacks.clear()

        # Close publisher socket
        if hasattr(self, "pub_socket"):
            try:
                self.pub_socket.close(0)
            except Exception as e:
                print(f"[{self.name}] Error closing PUB socket: {e}")

        # Terminate the context
        if hasattr(self, "ctx"):
            try:
                self.ctx.term()
            except Exception as e:
                print(f"[{self.name}] Error terminating ZMQ context: {e}")

        print(f"[{self.name}] Module stopped gracefully.")
