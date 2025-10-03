import logging
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
from typing import IO, Dict, Optional


def setup_handler(
    stream: Optional[IO] = None,
    filename: Optional[str] = None,
    log_queue: Optional[mp.Queue] = None,
    formatter: logging.Formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ),
) -> logging.Handler:
    if stream is not None:
        handler = logging.StreamHandler(stream)
    elif filename is not None:
        handler = logging.FileHandler(filename)
    elif log_queue is not None:
        return QueueHandler(log_queue)
    else:
        # Default: stdout
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    return handler


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    stream: Optional[IO] = None,
    filename: Optional[str] = None,
    log_queue: Optional[mp.Queue] = None,
) -> logging.Logger:
    """
    Creates and returns a logger with optional output:
    - log_queue (multiprocessing-safe queue, preferred for child processes)
    - stream (e.g., sys.stdout)
    - filename (log file)
    - defaults to stdout if none is given
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if log_queue:
        logger.propagate = False

    logger.handlers.clear()
    handler = setup_handler(stream, filename, log_queue)
    logger.addHandler(handler)

    return logger


class LevelFilter(logging.Filter):
    def __init__(self, root_level: int = logging.WARNING):
        self.root_level = root_level
        self.log_levels: Dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """the root level has priority over custom levels"""
        level = self.log_levels.get(record.name, self.root_level)

        return self.root_level <= record.levelno and level <= record.levelno

    def set_root_level(self, level: int) -> None:
        self.root_level = level

    def add_level(self, name: str) -> None:
        self.log_levels[name] = self.root_level

    def set_level(self, name: str, level: int) -> None:
        if name not in self.log_levels:
            raise ValueError(f"{name} has no linked log level")
        self.log_levels[name] = level

    def set_levels(self, level: int) -> None:
        self.set_root_level(level)
        for name in self.log_levels:
            self.set_level(name, level)

    def del_level(self, name: str) -> None:
        del self.log_levels[name]


def setup_log_listener(log_queue: mp.Queue, filter: logging.Filter) -> QueueListener:
    """
    Starts a central logging listener that reads LogRecords from a queue
    and emits them using normal loggers/handlers.
    """
    formatter = logging.Formatter(
        "[%(asctime)s] [%(processName)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = setup_handler(formatter=formatter)
    handler.addFilter(filter)

    listener = QueueListener(log_queue, handler)
    return listener
