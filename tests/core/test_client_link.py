"""Client.run() must end when the HuRI link dies, not wait on its senders."""

import asyncio
import json

import pytest
import websockets

from src.core.client import Client, ClientHook, ClientSender
from src.core.dataclasses.config import (
    ClientConfig,
    ClientHookConfig,
    ClientSenderConfig,
)
from src.core.events import BytesEvent
from src.core.interface import Interface


class BlockingSender(ClientSender[BytesEvent]):
    """Like the web interface's senders: waits forever for input."""

    output_type = BytesEvent

    async def input_loop(self, ws):
        await asyncio.Event().wait()


class RecordingHook(ClientHook[BytesEvent]):
    input_type = BytesEvent
    received = []

    async def hook(self, data):
        RecordingHook.received.append(data.data)


class FakeInterface(Interface):
    def __init__(self):
        super().__init__(singletton=None)

    def get_senders(self):
        return {"blocking": BlockingSender}

    def get_hooks(self):
        return {"recording": RecordingHook}


test_interface = FakeInterface()


def make_config(url: str) -> ClientConfig:
    return ClientConfig(
        user_id="t",
        huri_url=url,
        interface_path=f"{__name__}:test_interface",
        hooks={"h": ClientHookConfig(name="recording", topics=["audio.out"], args={})},
        senders={"s": ClientSenderConfig(name="blocking", topic="audio.in", args={})},
        modules={},
    )


async def fake_huri(close_code, close_reason, before_close=None):
    """A HuRI that accepts the handshake, optionally sends something, then
    closes the socket with the given code."""

    async def handler(ws):
        json.loads(await ws.recv())  # the ClientConfig
        await ws.send(json.dumps({"type": "session_init", "user_id": "t"}))
        if before_close is not None:
            await ws.send(before_close)
        await asyncio.sleep(0.05)
        await ws.close(code=close_code, reason=close_reason)

    return await websockets.serve(handler, "127.0.0.1", 0)


@pytest.mark.asyncio
async def test_abnormal_close_ends_run_with_error():
    server = await fake_huri(1011, "replica died")
    port = server.sockets[0].getsockname()[1]
    try:
        client = Client(make_config(f"ws://127.0.0.1:{port}"))
        with pytest.raises(websockets.ConnectionClosedError):
            await asyncio.wait_for(client.run(), timeout=5)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_clean_close_ends_run_quietly():
    server = await fake_huri(1000, "")
    port = server.sockets[0].getsockname()[1]
    try:
        client = Client(make_config(f"ws://127.0.0.1:{port}"))
        await asyncio.wait_for(client.run(), timeout=5)  # returns, no raise
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unexpected_json_is_skipped_not_fatal():
    RecordingHook.received.clear()
    topic = b"audio.out"
    binary = len(topic).to_bytes(2, "big") + topic + b"\x01\x02"

    async def handler(ws):
        json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "session_init", "user_id": "t"}))
        await ws.send(json.dumps({"type": "session_error", "error": "nope"}))
        await ws.send(binary)  # must still be delivered after the odd message
        await asyncio.sleep(0.05)
        await ws.close(code=1000)

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        client = Client(make_config(f"ws://127.0.0.1:{port}"))
        await asyncio.wait_for(client.run(), timeout=5)
        await asyncio.sleep(0)  # let the hook task run
        assert RecordingHook.received == [b"\x01\x02"]
    finally:
        server.close()
        await server.wait_closed()
