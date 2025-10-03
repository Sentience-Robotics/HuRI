import multiprocessing as mp
import signal
import time
from multiprocessing.synchronize import Event
from typing import Dict

from src.tools.logger import (LevelFilter, QueueListener, logging,
                              setup_log_listener, setup_logger)

from .event_router import EventRouter
from .module import Module


class ModuleManager:
    """Control Modules, Inter-Module communication and Logging"""

    def __init__(self, modules: Dict[str, Module]):
        self.modules: Dict[str, Module] = modules

        self.router_process: mp.Process = None
        self.processes: Dict[str, mp.Process] = {}
        self.stop_events: Dict[str, Event] = {}

        self.log_queue = mp.Queue()
        self.logger = setup_logger("HuRI", log_queue=self.log_queue)
        self.level_filter = LevelFilter(logging.DEBUG)
        self.log_listener: QueueListener = setup_log_listener(
            self.log_queue, self.level_filter
        )

    def _start_event_router(self) -> None:
        """Used to handle inter-module communication, though events"""
        if self.router_process and self.router_process.is_alive():
            return

        logger = setup_logger("EventRouter", log_queue=self.log_queue)
        self.router_process = mp.Process(
            target=lambda: EventRouter(logger).start(), daemon=True
        )
        self.level_filter.add_level("EventRouter")
        self.router_process.start()
        time.sleep(0.01)
        self.logger.info(f"Router started (PID={self.router_process.pid})")

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

    def start_module(self, name):
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
            args=(module, name, self.log_queue, stop_event),
            daemon=True,
        )
        self.processes[name] = p
        self.stop_events[name] = stop_event
        self.level_filter.add_level(name)

        p.start()
        self.logger.info(f"{name} ({type(module)}) started (PID={p.pid})")

    def stop_module(self, name):
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
            self.level_filter.del_level(name)

    def start(self):
        """Start event router and modules"""  # TODO config (also logs levels)
        self.log_listener.start()
        self._start_event_router()
        for name in self.modules:
            self.start_module(name)

    def stop_all(self):
        for name in list(self.processes.keys()):
            self.stop_module(name)
        if self.router_process and self.router_process.is_alive():
            self.logger.info("Stopping router...")
            self.router_process.terminate()
            self.router_process.join(timeout=5)
            self.level_filter.del_level("EventRouter")
            self.logger.info("Router stopped")

        self.log_listener.stop()

    def status(self):
        """Print status of all modules and router."""
        print("=== Module Status ===")
        if self.router_process:
            router_state = "alive" if self.router_process.is_alive() else "stopped"
            print(f"- Router: {router_state} (PID={self.router_process.pid})")
        else:
            print("- Router: not started")

        for name in self.modules:
            process = self.processes.get(name)
            if process:
                state = "alive" if process.is_alive() else "stopped"
                print(f"- {name}: {state} (PID={process.pid})")
            else:
                print(f"- {name}: stopped")
        print("=====================")

    def set_root_log_level(self, level: int) -> None:
        self.level_filter.set_root_level(level)

    def set_log_level(self, name: str, level: int) -> None:
        self.level_filter.set_level(name, level)

    def set_log_levels(self, name: str, level: int) -> None:
        self.level_filter.set_levels(level)

    def log_status(self) -> None:
        """Print status of all modules and router."""
        print("=== Log Status ===")
        print(f"Root level: {logging.getLevelName(self.level_filter.root_level)}")
        for name, lvl in self.level_filter.log_levels.items():
            print(f"- {name}: {logging.getLevelName(lvl)}")
        print("=====================")
