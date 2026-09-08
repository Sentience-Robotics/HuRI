"""Web interface: bridges a browser websocket to a HuRI :class:`Client` session.

This plays the same role for a browser that ``cli_interface.py`` plays for a
terminal: it defines the device/transport-specific :class:`ClientSender` and
:class:`ClientHook` subclasses, and owns the transport plumbing itself (a
FastAPI ``WebSocket`` route here, ``sounddevice``/``prompt_toolkit`` there)
rather than pushing that concern into ``src/core/client.py``.

Unlike the CLI (one process = one session), a web server handles many
concurrent browser connections. ``Interface.singletton`` is read exactly once,
synchronously, inside ``Client.__init__`` (see ``src/core/client.py``), so each
connection just needs its own value visible at that moment — there is no need
to plumb a session object through `Client`/`Interface`'s constructors. A
``contextvars.ContextVar`` gives exactly that: :func:`run_browser_session` sets
it to a fresh :class:`BrowserBridge` right before constructing this
connection's ``Client``, so every sender/hook that ``Client`` builds captures
the correct bridge — while ``CLIInterface`` (still a plain ``None`` attribute)
is completely untouched.

The website's own backend (Authelia/OIDC auth, static hosting, CORS — see
``HuRI_website_demo/backend/main.py``) is expected to authenticate the visitor
and then call :func:`run_browser_session` directly with the resolved
``user_id``. The ``router`` exposed here is only a no-auth convenience for
standalone testing of this interface.
"""

import asyncio
import base64
import contextvars
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.client import Client, ClientHook, ClientSender
from src.core.dataclasses.config import (
    ClientConfig,
    ClientHookConfig,
    ClientSenderConfig,
    ModuleConfig,
)
from src.core.events import BytesEvent
from src.core.interface import Interface
from src.modules.gesture.events import Motion
from src.modules.rag.events import RAGQuestion
from src.modules.speech_to_text.events import Transcript
from src.modules.text_to_speech.events import Audio, Token

logger = logging.getLogger("ray.serve")

# Module names this interface is willing to build a session from. Kept as a
# static allow-list instead of importing src.modules.modules.get_modules(),
# which would drag every module's heavy ML dependencies (torch, transformers,
# ray.serve deployment handles, ...) into whatever lightweight process hosts
# this interface's websocket route — that process only ever talks to a
# *remote* HuRI over a websocket, it never instantiates a Module itself.
KNOWN_MODULES = {"mic", "stt", "tag", "emo", "eag", "qag", "rag", "tts", "gesture"}

# Maps a module whose presence implies a hook is meaningful to the (hook
# config key, topic) it should subscribe to. HuRI's session handshake rejects
# a hook topic that no configured module actually produces
# (``HuRI._check_subscriptions`` in src/core/huri.py), so hooks are derived
# from the requested modules rather than always-on.
#
# Deliberately no "emo" -> "emotion" entry: "emotion" is EMO's raw per-chunk
# prosody read, emitted many times over one utterance. The one that matters
# for display is the aggregated, utterance-final read QAG links onto
# RAGQuestion.emotion, already carried by the "question" hook below — so
# there is exactly one path emotion data reaches the browser through.
_HOOK_TOPIC_BY_PRODUCER = {
    "rag": ("token", "token"),
    "tts": ("audio", "audio.out"),
    "gesture": ("motion", "motion"),
    "qag": ("question", "question"),
}
_HOOK_NAME_BY_TOPIC = {
    "token": "web_token",
    "audio.out": "web_audio",
    "motion": "web_motion",
    "question": "web_question",
}


@dataclass
class BrowserBridge:
    """Per-browser-connection state, threaded through ``Client`` via the
    ContextVar-backed ``WebInterface.singletton`` below.

    Senders pull browser-originated input off ``inbound`` queues (one per
    topic); hooks push HuRI-originated events onto ``outbound``, which a
    single writer task drains to the real websocket — Starlette websockets
    aren't safe for concurrent ``send_*`` calls from multiple hooks.
    """

    inbound: Dict[str, "asyncio.Queue[Any]"] = field(default_factory=dict)
    outbound: "asyncio.Queue[Dict[str, Any]]" = field(default_factory=asyncio.Queue)

    def queue_for(self, topic: str) -> "asyncio.Queue[Any]":
        return self.inbound.setdefault(topic, asyncio.Queue())

    async def send(self, payload: Dict[str, Any]) -> None:
        await self.outbound.put(payload)


_current_bridge: "contextvars.ContextVar[Optional[BrowserBridge]]" = (
    contextvars.ContextVar("huri_web_browser_bridge", default=None)
)


# ---------------------------------------------------------------------------
# Senders: browser -> HuRI
# ---------------------------------------------------------------------------


class WebAudioSender(ClientSender[BytesEvent]):
    """Relays raw mic PCM frames the browser already captured (HuRI/ATP.xlsx
    F5/F6 — voice activity detection and transcription)."""

    output_type = BytesEvent

    async def input_loop(self, ws):
        queue = self.singletton.queue_for(self.topic)
        while True:
            chunk: bytes = await queue.get()
            await self.send(ws, BytesEvent(data=chunk))


class WebTextSender(ClientSender[RAGQuestion]):
    """Typed text sent through the full RAG pipeline: the "rag.in" test case
    (HuRI/ATP.xlsx F9 — retrieve & augment)."""

    output_type = RAGQuestion

    async def input_loop(self, ws):
        queue = self.singletton.queue_for(self.topic)
        while True:
            text: str = await queue.get()
            await self.send(ws, RAGQuestion(Transcript(text=text, end=True), None))


class WebTokenSender(ClientSender[Token]):
    """Typed text injected straight at TTS/Gesture, bypassing RAG entirely:
    the "rag.out" test case (HuRI/ATP.xlsx F7/F8 — speech & gesture
    generation)."""

    output_type = Token

    async def input_loop(self, ws):
        queue = self.singletton.queue_for(self.topic)
        while True:
            text: str = await queue.get()
            await self.send(ws, Token(text=text, end=False))
            await self.send(ws, Token(text="", end=True))


# ---------------------------------------------------------------------------
# Hooks: HuRI -> browser
# ---------------------------------------------------------------------------


class WebTokenHook(ClientHook[Token]):
    """Streamed RAG answer text."""

    input_type = Token

    async def hook(self, data: Token):
        await self.singletton.send(
            {"type": "token", "text": data.text, "end": bool(data.end)}
        )


class WebAudioHook(ClientHook[Audio]):
    """Streamed TTS speech, base64-encoded for JSON transport."""

    input_type = Audio

    async def hook(self, data: Audio):
        pcm = np.ascontiguousarray(data.data, dtype="<f4")
        await self.singletton.send(
            {
                "type": "audio",
                "pts": data.pts,
                "sample_rate": data.sample_rate,
                "end": bool(data.end),
                "data": base64.b64encode(pcm.tobytes()).decode("ascii"),
            }
        )


class WebMotionHook(ClientHook[Motion]):
    """Streamed gesture frames. Forwarded as raw pose/expression/translation
    arrays — mapping these onto a specific 3D rig's bones/blendshapes is a
    presentation concern the website backend owns (see
    ``HuRI_website_demo/backend/pipeline.py``), not this interface."""

    input_type = Motion

    async def hook(self, data: Motion):
        await self.singletton.send(
            {
                "type": "motion",
                "pts": data.pts,
                "fps": data.fps,
                "poses": data.poses.tolist(),
                "expressions": data.expressions.tolist(),
                "trans": data.trans.tolist(),
            }
        )


class WebQuestionHook(ClientHook[RAGQuestion]):
    """The fully aggregated question right before RAG consumes it — already
    carries both the transcript and its linked emotion, so subscribing here
    satisfies HuRI/ATP.xlsx F9 ("full question" display) and F11 (transcript
    + linked emotion) with no changes to core HuRI."""

    input_type = RAGQuestion

    async def hook(self, data: RAGQuestion):
        emotion = data.emotion
        await self.singletton.send(
            {
                "type": "question",
                "text": data.transcript.text,
                "emotion": (
                    {
                        "label": emotion.label,
                        "confidence": emotion.confidence,
                        "scores": emotion.scores,
                    }
                    if emotion
                    else None
                ),
            }
        )


class WebInterface(Interface):
    """Browser-facing Interface. See module docstring for the ContextVar
    mechanism behind ``singletton``."""

    def __init__(self):
        super().__init__(singletton=None)

    @property
    def singletton(self) -> Optional[BrowserBridge]:
        return _current_bridge.get()

    @singletton.setter
    def singletton(self, value: Optional[BrowserBridge]) -> None:
        # Interface.__init__ assigns `self.singletton = singletton` once, with
        # None, which lands here. Real per-connection values are threaded in
        # via `_current_bridge.set(...)` in run_browser_session instead.
        pass

    def get_senders(self) -> Dict[str, Type[ClientSender]]:
        return {
            "web_audio": WebAudioSender,
            "web_text": WebTextSender,
            "web_token": WebTokenSender,
        }

    def get_hooks(self) -> Dict[str, Type[ClientHook]]:
        return {
            "web_token": WebTokenHook,
            "web_audio": WebAudioHook,
            "web_motion": WebMotionHook,
            "web_question": WebQuestionHook,
        }


web_interface = WebInterface()


# ---------------------------------------------------------------------------
# Browser-facing transport
# ---------------------------------------------------------------------------


def _build_client_config(
    handshake: Dict[str, Any], user_id: str
) -> Optional[ClientConfig]:
    """Turn a browser handshake into a ``ClientConfig``, or ``None`` if it
    doesn't describe a valid session.

    ``handshake["modules"]`` mirrors ``config/client_full.yaml``'s ``modules``
    block: ``{tag: {"name": <module name>, "args": {...}}}``. This is exactly
    what the frontend's Event Configuration modal edits, letting a tester pick
    any module combination (HuRI/ATP.xlsx F1/F2) — from a single ``rag``
    module (text-only) up to the full voice+gesture pipeline.
    """
    requested = handshake.get("modules")
    if not isinstance(requested, dict) or not requested:
        return None

    modules: Dict[str, ModuleConfig] = {}
    for tag, entry in requested.items():
        if not isinstance(entry, dict):
            return None
        name = entry.get("name")
        if name not in KNOWN_MODULES:
            return None
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            return None
        modules[tag] = ModuleConfig(name=name, args=args)

    present = {m.name for m in modules.values()}
    hooks: Dict[str, ClientHookConfig] = {}
    for producer, (hook_key, topic) in _HOOK_TOPIC_BY_PRODUCER.items():
        if producer in present:
            hooks[hook_key] = ClientHookConfig(
                name=_HOOK_NAME_BY_TOPIC[topic], topics=[topic], args={}
            )

    # Senders are always offered: a topic with no consumer in the requested
    # pipeline is simply a no-op publish (src/core/bus.py's EventGraph drops
    # events with zero subscribers), not a session-rejecting error. This is
    # what lets a tester target "token" directly (rag.out) even when rag/qag
    # aren't part of the session.
    senders: Dict[str, ClientSenderConfig] = {
        "audio": ClientSenderConfig(name="web_audio", topic="audio.in", args={}),
        "text": ClientSenderConfig(name="web_text", topic="question", args={}),
        "token": ClientSenderConfig(name="web_token", topic="token", args={}),
    }

    huri_url = os.environ.get("HURI_URL", "ws://localhost:8000/session")
    return ClientConfig(
        user_id=user_id,
        huri_url=huri_url,
        interface_path="src.interfaces.web_interface:web_interface",
        hooks=hooks,
        senders=senders,
        modules=modules,
    )


def describe_config(config: ClientConfig) -> Dict[str, Any]:
    """JSON-serializable summary of a resolved ``ClientConfig``, echoed back to
    the browser so the Event Configuration modal can render what's active."""
    return {
        "modules": {
            k: {"name": v.name, "args": v.args} for k, v in config.modules.items()
        },
        "senders": {
            k: {"name": v.name, "topic": v.topic, "args": v.args}
            for k, v in config.senders.items()
        },
        "hooks": {
            k: {"name": v.name, "topics": v.topics, "args": v.args}
            for k, v in config.hooks.items()
        },
    }


async def _pump_outbound(
    bridge: BrowserBridge,
    ws: WebSocket,
    transform_outbound: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> None:
    while True:
        payload = await bridge.outbound.get()
        if transform_outbound is not None:
            payload = transform_outbound(payload)
        await ws.send_json(payload)


async def _pump_inbound(bridge: BrowserBridge, ws: WebSocket) -> None:
    """Demux browser input: binary frames are mic PCM (always targets
    ``audio.in``); JSON frames are typed text tagged with which topic/event
    the tester picked in the Composer's event dropdown."""
    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect()

        raw_bytes = msg.get("bytes")
        if raw_bytes is not None:
            bridge.queue_for("audio.in").put_nowait(raw_bytes)
            continue

        raw_text = msg.get("text")
        if raw_text is None:
            continue
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            continue

        topic = "token" if data.get("topic") == "token" else "question"
        bridge.queue_for(topic).put_nowait(data.get("text", ""))


async def run_browser_session(
    ws: WebSocket,
    user_id: str,
    transform_outbound: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> None:
    """Drive one browser<->HuRI session end to end.

    Assumes ``ws`` is already accepted (``await ws.accept()``) — the caller
    (the website backend) owns auth and may need to send an error/close a
    connection before this ever gets involved.

    ``transform_outbound``, if given, is applied to every outbound message
    right before it's sent to the browser. This is the seam a website backend
    uses to reshape a ``"motion"`` message's raw pose/expression/translation
    arrays into whatever per-bone format its specific 3D asset's rig expects
    (see ``HuRI_website_demo/backend/pipeline.py``) — this interface only
    knows about HuRI's wire events, never about any particular character rig.

    Protocol:
      1. the browser sends one handshake JSON message: ``{"modules": {...}}``
         (see :func:`_build_client_config`);
      2. we reply ``{"type": "session_config", "config": {...}}`` with the
         resolved config (:func:`describe_config`);
      3. from then on it's the steady-state protocol: binary frames are mic
         PCM, JSON frames are ``{"topic": "question"|"token", "text": ...}``
         inbound, and outbound messages are ``{"type": "token"|"audio"|
         "motion"|"question", ...}`` (see the hooks above).
    """
    try:
        handshake = await ws.receive_json()
    except Exception:
        await ws.close(code=1002)
        return

    config = _build_client_config(handshake, user_id=user_id)
    if config is None:
        await ws.send_json(
            {"type": "error", "message": "invalid or missing session config"}
        )
        await ws.close(code=1008)
        return

    bridge = BrowserBridge()
    token = _current_bridge.set(bridge)
    try:
        client = Client(config=config)
    except Exception as exc:  # e.g. an unresolvable module combination
        _current_bridge.reset(token)
        logger.exception("web_interface: failed to build Client")
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close(code=1011)
        return
    _current_bridge.reset(token)

    await ws.send_json({"type": "session_config", "config": describe_config(config)})

    client_task = asyncio.create_task(client.run())
    writer_task = asyncio.create_task(
        _pump_outbound(bridge, ws, transform_outbound=transform_outbound)
    )

    inbound_task = asyncio.create_task(_pump_inbound(bridge, ws))
    try:
        # Race the browser's own input against the HuRI-side session dying
        # underneath us (e.g. a requested module isn't deployed on this
        # instance and HuRI closes the socket after "session_error"). Without
        # this, client_task's exception would only surface at GC time as an
        # unretrieved-task warning, and the browser would just see nothing
        # happen — indistinguishable from a slow reply.
        done, _ = await asyncio.wait(
            {client_task, inbound_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if client_task in done and not client_task.cancelled():
            client_exc = client_task.exception()
            if client_exc is not None:
                logger.warning("web_interface: HuRI session ended: %s", client_exc)
                try:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": f"HuRI session ended: {client_exc}",
                        }
                    )
                except Exception:
                    pass
        if inbound_task in done and not inbound_task.cancelled():
            inbound_exc = inbound_task.exception()
            if inbound_exc is not None and not isinstance(
                inbound_exc, WebSocketDisconnect
            ):
                logger.exception("web_interface: session error", exc_info=inbound_exc)
    finally:
        client_task.cancel()
        inbound_task.cancel()
        writer_task.cancel()
        await asyncio.gather(
            client_task, inbound_task, writer_task, return_exceptions=True
        )


router = APIRouter()


@router.websocket("/ws")
async def _standalone_ws(ws: WebSocket) -> None:
    """No-auth entrypoint for standalone testing of this interface only. The
    website backend does not mount this route — it wraps
    :func:`run_browser_session` behind Authelia/magic-link auth instead (see
    ``HuRI_website_demo/backend/main.py``)."""
    await ws.accept()
    user_id = ws.query_params.get("user_id") or str(uuid.uuid4())
    await run_browser_session(ws, user_id=user_id)
