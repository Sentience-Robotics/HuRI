import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import Dict
import sys
from src.tools.logger import (
    LevelFilter,
    QueueListener,
    logging,
    setup_log_listener,
    setup_logger,
)

from .zmq.control_channel import Router
from .zmq.event_proxy import EventProxy
from .zmq.log_channel import LogPuller


@dataclass
class RouterConfig:
    port: int


@dataclass
class EventProxyConfig:
    xsub: int
    xpub: int


@dataclass
class LogPullerConfig:
    port: int


@dataclass
class HuriConfig:
    hostname: str
    router: RouterConfig
    event_proxy: EventProxyConfig
    log_puller: LogPullerConfig

    @classmethod
    def from_dict(cls, raw: dict):
        return cls(
            hostname=raw["hostname"],
            router=RouterConfig(**raw["router"]),
            event_proxy=EventProxyConfig(**raw["event-proxy"]),
            log_puller=LogPullerConfig(**raw["log-puller"]),
        )


class HuRI:
    """Wait for Agent to connect, handle module communication and Logging"""

    def __init__(self, config: HuriConfig) -> None:
        self.router = Router(config.hostname, config.router.port)
        self.event_proxy = EventProxy(
            config.hostname, "", config.event_proxy.xpub, config.event_proxy.xsub
        )
        self.log_channel = LogPuller(config.hostname, config.log_puller.port)

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
        self._start_log_channel()
        self._start_router()
        self._start_event_proxy()

        if not sys.stdin.isatty():
            threading.Event().wait()
            return

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
