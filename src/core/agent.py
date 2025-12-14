import multiprocessing as mp
import signal
import sys
import threading
import time
from multiprocessing.synchronize import Event
from typing import Dict, List

import zmq

from src.tools.logger import (
    LevelFilter,
    QueueListener,
    logging,
    setup_log_listener,
    setup_logger,
)

from .module import Module
from .zmq.event_proxy import EventProxy, ZMQEventPorts
from .zmq.log_channel import LogPusher, ZMQLogPort
from .zmq.control_channel import Command, Dealer, ZMQRouterPort

# class AgentConfig:
#     huri_hostname: str
#     huri_event_router_ports: ZMQEventPorts
#     event_forwarder_ports: ZMQEventPorts
#     log_level: int = logging.INFO


class Agent:
    """Control Modules and communication with HuRI"""

    def __init__(self, modules: Dict[str, Module]) -> None:
        self.modules: Dict[str, Module] = modules

        self.processes: Dict[str, mp.Process] = {}
        self.stop_events: Dict[str, Event] = {}

        self.threads: Dict[str, threading.Thread] = {}

        self.log_pusher = LogPusher("tcp://localhost:8008")

        self.dealer = Dealer(
            port=ZMQRouterPort(router="3000"),
            executor=self._command_handler,
            logger=setup_logger("Dealer", log_queue=self.log_pusher.log_queue),
        )

        self.up_proxy = EventProxy(
            ports=ZMQEventPorts(xpub="5555", xsub="6666"),
            logger=setup_logger("UpProxy", log_queue=self.log_pusher.log_queue),
        )
        self.down_proxy = EventProxy(
            ports=ZMQEventPorts(xpub="6665", xsub="5556"),
            logger=setup_logger("DownProxy", log_queue=self.log_pusher.log_queue),
        )

        self.logger = setup_logger(
            f"Agent {self.dealer.identity}", log_queue=self.log_pusher.log_queue
        )

    def _command_handler(self, command: Command) -> bool:
        match command.cmd:
            case "START":
                return self.start_module(*command.args)
            case "STOP":
                return self.stop_module(*command.args)
            case "STATUS":
                return self.status()
            case _:
                return False  # todo log

    @staticmethod
    def _start_module(
        module: Module, name: str, log_queue: mp.Queue, stop_event: Event
    ) -> None:
        """Helper function to start module in child process."""
        logger = setup_logger(name, log_queue=log_queue)
        module.set_custom_logger(logger)

        def handle_sigint(signum, frame):
            logger.info(f"Ctrl+C ignored in child module")

        signal.signal(signal.SIGINT, handle_sigint)

        module.start_module(stop_event=stop_event)

    def start_module(self, name) -> None:
        """Check if module is registered and not already running, and start a child process."""
        if name not in self.modules:
            self.logger.warning(
                f"{name} is not in the registered Modules: {self.modules.keys()}"
            )
            return
        if name in self.processes:
            self.logger.warning(
                f"{name} is already running (PID={self.processes[name].pid})"
            )
            return

        module = self.modules[name]
        stop_event = mp.Event()
        p = mp.Process(
            target=self._start_module,
            args=(module, name, self.log_pusher.log_queue, stop_event),
            daemon=True,
        )
        self.processes[name] = p
        self.stop_events[name] = stop_event
        self.log_pusher.level_filter.add_level(name)

        p.start()
        self.logger.info(f"{name} ({type(module)}) started (PID={p.pid})")

    def stop_module(self, name) -> None:
        if name in self.processes:
            self.logger.info(f"Stopping {name}...")
            self.stop_events[name].set()
            self.processes[name].join(timeout=5)
            if self.processes[name].is_alive():
                self.logger.warning(f"{name} did not stop in time, killing")
                self.processes[name].kill()
            self.logger.info(f"{name} stopped")
            del self.processes[name]
            del self.stop_events[name]
            self.log_pusher.level_filter.del_level(name)

    def stop_all(self) -> None:
        for name in list(self.processes.keys()):
            self.stop_module(name)

        self.dealer.stop()
        self.up_proxy.stop()
        self.down_proxy.stop()
        for name, thread in self.threads.items():
            self.logger.info(f"Stopping {name} thread...")
            thread.join(timeout=5)
            self.logger.info(f"{name} thread stopped")
            self.log_pusher.level_filter.del_level(name)

        self.log_pusher.stop()
        print("Fully stopped")

    def status(self) -> None:
        """Print status of all modules and router."""
        print("=== Module Status ===")
        # if self.router_process:
        #     router_state = "alive" if self.router_process.is_alive() else "stopped"
        #     print(f"- Router: {router_state} (PID={self.router_process.pid})")
        # else:
        #     print("- Router: not started")

        for name in self.modules:
            process = self.processes.get(name)
            if process:
                state = "alive" if process.is_alive() else "stopped"
                print(f"- {name}: {state} (PID={process.pid})")
            else:
                print(f"- {name}: stopped")
        print("=====================")

    def set_root_log_level(self, level: int) -> None:
        self.log_pusher.level_filter.set_root_level(level)

    def set_log_level(self, name: str, level: int) -> None:
        self.log_pusher.level_filter.set_level(name, level)

    def set_log_levels(self, level: int) -> None:
        self.log_pusher.level_filter.set_levels(level)

    # def log_status(self) -> None:
    #     """Print status of all modules and router."""
    #     print("=== Log Status ===")
    #     print(f"Root level: {logging.getLevelName(self.level_filter.root_level)}")
    #     for name, lvl in self.level_filter.log_levels.items():
    #         print(f"- {name}: {logging.getLevelName(lvl)}")
    #     print("=====================")

    def _connect_to_huri(self) -> None:
        self.log_pusher.level_filter.add_level("Dealer")
        self.threads["Dealer"] = threading.Thread(target=self.dealer.start)
        self.threads["Dealer"].start()

    def _start_event_proxies(self) -> None:
        """Used to handle inter-module communication, though events"""
        self.log_pusher.level_filter.add_level("UpProxy")
        self.log_pusher.level_filter.add_level("DownProxy")
        self.threads["UpProxy"] = threading.Thread(
            target=self.up_proxy.start, args=[True, False]
        )
        self.threads["DownProxy"] = threading.Thread(
            target=self.down_proxy.start, args=[False, True]
        )

        self.threads["UpProxy"].start()
        self.threads["DownProxy"].start()

    def run(self) -> None:
        """Start event router and modules"""  # TODO config (also logs levels)

        # def handle_sigint(signum, frame):
        #     self.logger.info(f"Ctrl+C detected, stopping...")
        #     self.stop_all()

        try:
            self.log_pusher.start()
            self._connect_to_huri()
            self._start_event_proxies()
        except Exception as e:
            self.logger.error(e)
            return

        for name in self.modules:
            self.start_module(name)

        threading.Event().wait()

        # def handle_sigint(sig, frame):
        #     shutdown_event.set()

        # shutdown_event = threading.Event()
        # signal.signal(signal.SIGINT, handle_sigint)
        # shutdown_event.wait()
        # signal.signal(signal.SIGINT, handle_sigint)
