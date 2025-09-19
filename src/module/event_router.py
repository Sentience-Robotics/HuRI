from typing import Optional

import zmq

from src.tools.logger import logging, setup_logger

XPUB_ENDPOINT = "tcp://localhost:5555"
XSUB_ENDPOINT = "tcp://localhost:5556"


class EventRouter:
    def __init__(self, logger: Optional[logging.Logger] = setup_logger("EventRouter")):

        self.ctx = zmq.Context()
        self.xpub = self.ctx.socket(zmq.XPUB)
        self.xsub = self.ctx.socket(zmq.XSUB)
        self.xpub.bind(XPUB_ENDPOINT)
        self.xsub.bind(XSUB_ENDPOINT)

        self.logger = logger

    def start(self):
        try:
            zmq.proxy(self.xsub, self.xpub)
        except KeyboardInterrupt:
            self.logger.info("Ctrl+C pressed, exiting cleanly")
        except Exception as e:
            self.logger.error(e)
