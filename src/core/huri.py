import multiprocessing as mp
import threading
import time
from typing import Dict

from src.tools.logger import (
    LevelFilter,
    QueueListener,
    logging,
    setup_log_listener,
    setup_logger,
)

from .zmq.event_proxy import EventProxy, ZMQEventPorts
from .zmq.log_channel import LogPuller, ZMQLogPort
from .zmq.control_channel import Router, ZMQRouterPort


class HuRIConfig:
    register_channel_ports: ZMQEventPorts
    event_channel_ports: ZMQEventPorts
    log_channel_port: ZMQEventPorts
    log_level: int = logging.INFO


class HuRI:
    """Wait for Agent to connect, handle module communication and Logging"""

    def __init__(self) -> None:
        self.router = Router(ZMQRouterPort(router="3000"))
        self.event_proxy = EventProxy(ZMQEventPorts(xpub="5556", xsub="5555"))
        self.log_channel = LogPuller(ZMQLogPort("8008"))

        self.threads: Dict[str, threading.Thread] = {}

        self.logger = setup_logger("HuRI")

    def _start_router(self) -> None:
        """Used to handle Agent registration and control"""
        self.threads["Router"] = threading.Thread(target=self.router.start)
        self.threads["Router"].start()

    def _start_event_proxy(self) -> None:
        """Used to handle inter-module communication, though events"""
        self.threads["EventProxy"] = threading.Thread(
            target=self.event_proxy.start, args=[False, False]
        )
        self.threads["EventProxy"].start()

    def _start_log_channel(self) -> None:
        """Used to handle Agent registration and control"""
        self.threads["LogChannel"] = threading.Thread(target=self.log_channel.start)
        self.threads["LogChannel"].start()

    def run(self) -> None:
        # self.log_listener.start()
        self._start_log_channel()
        self._start_router()
        self._start_event_proxy()
        # self._start_log_channel()

        from src.core.shell import RobotShell

        RobotShell(self).cmdloop()

    def stop(self) -> None:
        self.router.stop()
        self.event_proxy.stop()
        self.log_channel.stop()
        for name, thread in self.threads.items():
            self.logger.info(f"Stopping {name} thread...")
            thread.join(timeout=5)
            self.logger.info(f"{name} thread stopped")
