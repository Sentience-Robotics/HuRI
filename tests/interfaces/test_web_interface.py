"""Browser -> bridge demux (web_interface._route_inbound)."""

import json

from src.interfaces.web_interface import END_OF_SPEECH, BrowserBridge, _route_inbound


def drain(queue):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def text(payload) -> dict:
    return {"type": "websocket.receive", "text": json.dumps(payload)}


def test_binary_frames_are_mic_audio():
    bridge = BrowserBridge()
    _route_inbound(bridge, {"type": "websocket.receive", "bytes": b"\x01\x02"})
    assert drain(bridge.queue_for("audio.in")) == [b"\x01\x02"]


def test_mic_closed_marker_lands_behind_the_audio_on_the_same_queue():
    bridge = BrowserBridge()
    _route_inbound(bridge, {"type": "websocket.receive", "bytes": b"\x01\x02"})
    _route_inbound(bridge, text({"topic": "audio.in", "end": True}))
    assert drain(bridge.queue_for("audio.in")) == [b"\x01\x02", END_OF_SPEECH]
    assert END_OF_SPEECH == b""
    # an empty binary frame is the same marker
    _route_inbound(bridge, {"type": "websocket.receive", "bytes": b""})
    assert drain(bridge.queue_for("audio.in")) == [END_OF_SPEECH]


def test_audio_in_json_without_end_is_ignored():
    bridge = BrowserBridge()
    _route_inbound(bridge, text({"topic": "audio.in", "end": False}))
    _route_inbound(bridge, text({"topic": "audio.in", "text": "not audio"}))
    assert not bridge.inbound  # no queue was even created
    assert drain(bridge.queue_for("question")) == []


def test_typed_text_targets_the_picked_topic():
    bridge = BrowserBridge()
    _route_inbound(bridge, text({"topic": "question", "text": "hi"}))
    _route_inbound(bridge, text({"topic": "token", "text": "say this"}))
    _route_inbound(bridge, text({"text": "default"}))  # no topic -> question
    assert drain(bridge.queue_for("question")) == ["hi", "default"]
    assert drain(bridge.queue_for("token")) == ["say this"]


def test_garbage_is_dropped():
    bridge = BrowserBridge()
    _route_inbound(bridge, {"type": "websocket.receive", "text": "{not json"})
    _route_inbound(bridge, {"type": "websocket.receive", "text": "[1, 2]"})
    _route_inbound(bridge, {"type": "websocket.receive"})
    assert not bridge.inbound


# ---------------------------------------------------------------------------
# End to end through the bridge: browser marker -> Client -> HuRI wire frame
# ---------------------------------------------------------------------------

import asyncio
import struct

import pytest
import websockets

from src.interfaces.web_interface import run_browser_session


class FakeBrowserSocket:
    """Just enough of a Starlette WebSocket for run_browser_session."""

    def __init__(self, inbound):
        self._inbound = list(inbound)
        self.sent = []
        self.closed = None

    async def receive_json(self):
        return self._inbound.pop(0)

    async def receive(self):
        if self._inbound:
            return self._inbound.pop(0)
        await asyncio.sleep(0.2)  # give the senders time to relay
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed = code


@pytest.mark.asyncio
async def test_mic_closed_marker_reaches_huri_as_empty_audio_in_frame(monkeypatch):
    frames = []

    async def handler(ws):
        json.loads(await ws.recv())  # ClientConfig
        await ws.send(json.dumps({"type": "session_init", "user_id": "t"}))
        try:
            async for msg in ws:
                frames.append(msg)
        except websockets.ConnectionClosed:
            pass

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("HURI_URL", f"ws://127.0.0.1:{port}/session")
    try:
        ws = FakeBrowserSocket(
            [
                {"modules": {"mic": {"name": "mic", "args": {}}}},
                {"type": "websocket.receive", "bytes": b"\x01\x02"},
                {"type": "websocket.receive", "bytes": b""},  # what the ATP frontend sends
            ]
        )
        await asyncio.wait_for(run_browser_session(ws, user_id="t"), timeout=5)
    finally:
        server.close()
        await server.wait_closed()

    assert ws.sent[0]["type"] == "session_config"
    topic = b"audio.in"
    header = struct.pack(">H", len(topic)) + topic
    # HuRI's receive_loop splits [len][topic][payload]: the marker is the
    # audio.in packet with no payload, right behind the frame it terminates.
    assert frames == [header + b"\x01\x02", header]
