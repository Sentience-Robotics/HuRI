import json
import signal
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import zmq

from src.tools.logger import (
    LevelFilter,
    QueueListener,
    logging,
    mp,
    setup_log_listener,
    setup_logger,
)


def record_to_dict(record: logging.LogRecord) -> Dict[str, Any]:
    return {
        "name": record.name,
        "levelno": record.levelno,
        "levelname": record.levelname,
        "message": record.getMessage(),
        "created": record.created,
        "asctime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
        "process": record.process,
        "processName": record.processName,
        "thread": record.thread,
        "threadName": record.threadName,
        "module": record.module,
        "filename": record.filename,
        "pathname": record.pathname,
        "lineno": record.lineno,
        "funcName": record.funcName,
    }


def dict_to_record(data: Dict[str, Any]) -> logging.LogRecord:
    record = logging.LogRecord(
        name=data["name"],
        level=data["levelno"],
        pathname=data["pathname"],
        lineno=data["lineno"],
        msg=data["message"],
        args=(),
        exc_info=None,
        func=data["funcName"],
    )

    # Restore metadata
    record.created = data["created"]
    record.process = data["process"]
    record.processName = data["processName"]
    record.thread = data["thread"]
    record.threadName = data["threadName"]
    record.module = data["module"]
    record.filename = data["filename"]

    return record


class LogPuller:
    def __init__(
        self,
        hostname: str,
        port: int,
        logger: Optional[logging.Logger] = setup_logger("LogPuller"),
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.pull = self.ctx.socket(zmq.PULL)

        self.hostname = hostname
        self.port = port

        self.logger = logger or logging.getLogger(__name__)

    def start(self) -> None:
        self.pull.bind(f"tcp://{self.hostname}:{self.port}")

        self.logger.info("started")
        while True:
            payload = self.pull.recv()

            self.logger.handle(dict_to_record(json.loads(payload.decode())))

    def stop(self) -> None:
        self.pull.close()


class LogPusher:
    class LogPusherHandler(logging.Handler):
        def __init__(
            self,
            hostname: str,
            port: int,
        ):
            super().__init__()
            self.ctx = zmq.Context.instance()
            self.socket = self.ctx.socket(zmq.PUSH)

            self.hostname = hostname
            self.port = port

        def emit(self, record: logging.LogRecord) -> None:
            try:
                payload = json.dumps(record_to_dict(record)).encode()
                self.socket.send(payload)
            except Exception:
                self.handleError(record)
            except Exception:
                self.handleError(record)

        def start(self) -> None:
            self.socket.connect(f"tcp://{self.hostname}:{self.port}")

        def stop(self) -> None:
            self.socket.close()

    def __init__(
        self,
        hostname: str,
        port: int,
    ):

        self.log_queue = mp.Queue()

        self.log_handler = self.LogPusherHandler(hostname, port)
        self.level_filter = LevelFilter(logging.DEBUG)
        self.log_listener: QueueListener = setup_log_listener(
            self.log_queue, self.level_filter, self.log_handler
        )

        self.logger = setup_logger("LogPusher", log_queue=self.log_queue)

    def start(self) -> None:
        self.log_handler.start()
        self.log_listener.start()

    def stop(self):
        self.logger.info("stopping")
        time.sleep(0.2)
        self.log_listener.stop()
        self.log_handler.stop()
