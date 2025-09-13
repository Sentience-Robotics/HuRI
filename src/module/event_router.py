import zmq

XPUB_ENDPOINT = "tcp://localhost:5555"
XSUB_ENDPOINT = "tcp://localhost:5556"


class EventRouter:
    def __init__(self):

        self.ctx = zmq.Context()
        self.xpub = self.ctx.socket(zmq.XPUB)
        self.xsub = self.ctx.socket(zmq.XSUB)
        self.xpub.bind(XPUB_ENDPOINT)
        self.xsub.bind(XSUB_ENDPOINT)

    def start(self):
        print("EventRouter started")
        zmq.proxy(self.xsub, self.xpub)
