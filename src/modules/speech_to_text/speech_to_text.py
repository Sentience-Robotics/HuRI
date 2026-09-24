import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ray import serve
from ray.serve import handle

from src.core.module import ModuleWithHandle

from .events import Transcript, Voice

logger = logging.getLogger("ray.serve")

_MODEL_PATH = os.environ.get("HURI_STT_MODEL_PATH", "base")
_NUM_WORKERS = int(os.environ.get("HURI_STT_NUM_WORKERS", "2"))

# faster-whisper takes raw arrays at 16 kHz only; used for the duration logs.
_WHISPER_RATE = 16000


@serve.deployment(name="STT", max_ongoing_requests=8)
class STTDeployment:
    """faster-whisper model wrapper.

    Holds the WhisperModel and runs transcription on its own Ray Serve actor,
    off the HuRI master actor — model load and GPU inference no longer block the
    websocket ingress / per-session router. Pinned to a GPU worker via
    ray_actor_options in the Serve config (see deploy values.yaml).

    Stateless across calls: the per-session buffering lives in the STT module,
    so this deployment is shared across all sessions.

    Two call flavours (see STT below): a *partial* pass over a short sliding
    window (greedy, fast, display-only) and a *final* pass over the whole
    utterance (beam search + Silero VAD filter, the text that reaches RAG).

    :model: path to (or size name of) the faster-whisper model. Defaults to the
        HURI_STT_MODEL_PATH env var, falling back to "base".
    :device: "cpu", "cuda", or "auto".
    :compute_type: e.g. "int8", "float16", or "auto".
    """

    def __init__(
        self,
        model: str = _MODEL_PATH,
        device: str = "auto",
        compute_type: str = "auto",
        num_workers: int = _NUM_WORKERS,
    ):
        from faster_whisper import WhisperModel

        # num_workers lets CTranslate2 service several transcriptions at once on
        # this single replica — each "worker" is an independent inference slot
        # over the shared (read-only) weights. Combined with the thread pool
        # below, N sessions are transcribed concurrently while staying fully
        # independent: no per-call client state is ever held here (the
        # per-session buffers live in the STT module).
        self.model_faster = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )
        # Run the *blocking* faster-whisper call off the actor's asyncio loop.
        # transcribe() used to be an async method that called the synchronous
        # model inline, blocking the replica's event loop for the whole inference
        # — serialising every session on the replica and stalling Serve health
        # checks. Offloading to a thread pool (sized to num_workers) keeps the
        # loop free to accept other sessions' requests while inference runs.
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="stt-transcribe"
        )

    async def transcribe(
        self, audio: np.ndarray, language: str = "en", final: bool = False
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._transcribe_sync, audio, language, final
        )

    def _transcribe_sync(self, audio: np.ndarray, language: str, final: bool) -> str:
        t0 = time.monotonic()
        if final:
            # Whole utterance, once per turn: afford beam search, and let the
            # bundled Silero VAD drop the pauses MIC now forwards and the
            # trailing silence — that is what stops Whisper from inventing
            # "Thank you." / "Thanks for watching." over non-speech.
            segments, _ = self.model_faster.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,
            )
        else:
            # Sliding-window partial, ~1/s while the user talks: greedy and
            # timestamp-free, it only has to be fast enough to keep up.
            segments, _ = self.model_faster.transcribe(
                audio,
                language=language,
                beam_size=1,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.log(
            logging.INFO if final else logging.DEBUG,
            "[STT.deploy] final=%s audio=%.2fs took=%.0fms",
            final,
            len(audio) / _WHISPER_RATE,
            1000 * (time.monotonic() - t0),
        )
        return text


@dataclass
class _Turn:
    """Snapshot of one utterance, handed to the final pass. Owns its lists."""

    turn_id: int
    frames: List[np.ndarray]
    samples: int
    partials: List[str]
    t_open: float


class STT(ModuleWithHandle):
    """STT Module

    Transcribe voice using Faster_Whisper.

    Holds the per-session buffers and delegates the actual transcription to a
    handle-backed STTDeployment, so the Whisper model runs off the HuRI master
    node.

    input: voice,
    output: transcript

    Contract with TAG:
      * ``Transcript(end=False)`` — best-effort partial from a short sliding
        window while the user talks. For live display only.
      * ``Transcript(end=True)`` — the COMPLETE text of the utterance, from one
        whole-utterance pass run when MIC sends ``Voice(None)``. This is the
        text that becomes the question.

    Concurrency: the bus spawns one ``process()`` task per Voice frame, so
    calls on this instance interleave at every ``await``. All per-turn state
    below is therefore mutated ONLY in await-free sections — ``_close_turn``
    snapshots and rebinds the turn synchronously before anything is awaited,
    and a partial that resumes after its turn was closed recognises that from
    ``turn_id`` and drops its result. That is what makes the end marker
    impossible to lose (the old shared ``self.silence`` flag could be flipped
    back by the next frame while a transcription was in flight, which left the
    turn open forever). ``_final_lock`` is not state protection: it just keeps
    ``end=True`` transcripts in turn order and one final in flight per session.

    :language: language spoken in the audio. It should be a language code such
        as "en" or "fr".
    :sample_rate: sample rate of the received voice audio.
    :block_duration: accepted for config compatibility; buffering is done in
        samples, so it is not needed here.
    :transcribe_window: duration of audio per partial transcription (in s).
    :transcribe_step: overlap between consecutive partial windows (in s).
    :max_final_duration: at most this much audio (the tail) goes to the final
        pass (in s). MIC's ``max_utterance_duration`` should stay below it.
    :partial_timeout: give up on a partial after this long (in s).
    :final_timeout: give up on the final pass after this long and fall back to
        the joined partials (in s).
    :final_latency_warn: log the final at WARNING when it took longer (in s).
    :enable_partials: set False to skip the sliding window entirely.
    """

    _handle_cls = STTDeployment
    input_type = "voice"
    output_type = "transcript"

    def __init__(
        self,
        _handle: handle.DeploymentHandle,
        language: str = "en",
        sample_rate: int = 16000,
        block_duration: float = 0.030,  # s, unused (see docstring)
        transcribe_window: float = 2.0,  # s
        transcribe_step: float = 1.0,  # s
        max_final_duration: float = 30.0,  # s
        partial_timeout: float = 4.0,  # s
        final_timeout: float = 20.0,  # s
        final_latency_warn: float = 1.5,  # s
        enable_partials: bool = True,
        **kwargs,
    ):
        super().__init__(_handle=_handle, **kwargs)

        self.language = language
        self.sample_rate = sample_rate

        self.window_samples = max(1, int(transcribe_window * sample_rate))
        overlap = min(max(transcribe_step, 0.0), transcribe_window)
        self.slide_samples = int((transcribe_window - overlap) * sample_rate)
        if self.slide_samples <= 0:
            self.slide_samples = self.window_samples
        self.max_final_samples = max(1, int(max_final_duration * sample_rate))

        self.partial_timeout = partial_timeout
        self.final_timeout = final_timeout
        self.final_latency_warn = final_latency_warn
        self.enable_partials = enable_partials

        # --- per-turn state: mutate ONLY in await-free sections (see class doc)
        self.turn_id: int = 1  # numbered like MIC's turns, so logs line up
        # Samples not yet consumed by the partial window.
        self.window_buf: np.ndarray = np.zeros(0, dtype=np.float32)
        # Every frame of the current turn (tail-capped at max_final_samples).
        self.utterance: List[np.ndarray] = []
        self.utterance_samples: int = 0
        self.partials: List[str] = []
        self._t_open: Optional[float] = None
        # One partial in flight at a time.
        self.running: bool = False
        # FIFO: finals are emitted in turn order, one per session at a time.
        self._final_lock: asyncio.Lock = asyncio.Lock()

    async def process(self, voice: Voice) -> Optional[Transcript]:
        if voice.data is None:
            # End of utterance. Close the turn SYNCHRONOUSLY — nothing may be
            # awaited before this, or a frame of the next utterance could slip
            # in and be mistaken for a continuation.
            turn = self._close_turn()
            if turn is None:
                logger.info("[STT] end marker on empty turn, ignored")
                return None
            return await self._finalize(turn)

        frame = np.asarray(voice.data, dtype=np.float32).reshape(-1)
        if self._t_open is None:
            self._t_open = time.monotonic()
        self.utterance.append(frame)
        self.utterance_samples += len(frame)
        # Bounded memory even if MIC never closes the turn: keep the tail.
        while self.utterance_samples > self.max_final_samples and len(self.utterance) > 1:
            self.utterance_samples -= len(self.utterance.pop(0))
        self.window_buf = np.concatenate((self.window_buf, frame))

        if (
            not self.enable_partials
            or self.running
            # A pending final gets the inference slot; partials can wait.
            or self._final_lock.locked()
            or len(self.window_buf) < self.window_samples
        ):
            return None

        # Partials are display-only: if inference fell behind real time, skip
        # ahead to the freshest window rather than transcribing stale audio.
        if len(self.window_buf) > self.window_samples + self.slide_samples:
            logger.debug(
                "[STT] turn %d partial backlog %.1fs, skipping ahead",
                self.turn_id,
                len(self.window_buf) / self.sample_rate,
            )
            self.window_buf = self.window_buf[-self.window_samples :]

        self.running = True
        tid = self.turn_id
        audio = self.window_buf[: self.window_samples]
        try:
            text = await self._transcribe(audio, final=False, timeout=self.partial_timeout)
        except Exception as e:
            logger.warning("[STT] turn %d partial failed: %r", tid, e)
            return None
        finally:
            self.running = False

        if tid != self.turn_id:
            logger.debug("[STT] turn %d partial completed after close, dropped", tid)
            return None
        self.window_buf = self.window_buf[self.slide_samples :]
        if text:
            self.partials.append(text)
        return Transcript(text, False)

    def _close_turn(self) -> Optional[_Turn]:
        """Snapshot the current turn and start a fresh one. Synchronous."""
        if not self.utterance:
            return None
        turn = _Turn(
            self.turn_id,
            self.utterance,
            self.utterance_samples,
            self.partials,
            self._t_open or time.monotonic(),
        )
        # Rebind — never .clear() — the snapshot owns the old containers.
        self.turn_id += 1
        self.utterance = []
        self.utterance_samples = 0
        self.partials = []
        self.window_buf = np.zeros(0, dtype=np.float32)
        self._t_open = None
        logger.info(
            "[STT] turn %d closed: %.2fs audio, %d partials",
            turn.turn_id,
            turn.samples / self.sample_rate,
            len(turn.partials),
        )
        return turn

    async def _finalize(self, turn: _Turn) -> Transcript:
        fallback = " ".join(turn.partials).strip()
        t0 = time.monotonic()
        async with self._final_lock:
            queued_ms = 1000 * (time.monotonic() - t0)
            try:
                audio = np.concatenate(turn.frames)
                text = await self._transcribe(audio, final=True, timeout=self.final_timeout)
            except Exception as e:
                logger.warning(
                    "[STT] turn %d final failed (%r), falling back to %d partials",
                    turn.turn_id,
                    e,
                    len(turn.partials),
                )
                text = fallback
        total_ms = 1000 * (time.monotonic() - t0)
        log = logger.warning if total_ms > 1000 * self.final_latency_warn else logger.info
        log(
            "[STT] turn %d final: audio=%.2fs queued=%.0fms total=%.0fms text=%r",
            turn.turn_id,
            turn.samples / self.sample_rate,
            queued_ms,
            total_ms,
            text,
        )
        # Always end=True, even when empty: TAG/QAG close the turn on it.
        return Transcript(text, True)

    async def _transcribe(self, audio: np.ndarray, final: bool, timeout: float) -> str:
        response = self._handle.transcribe.remote(audio, self.language, final=final)
        try:
            return await asyncio.wait_for(response, timeout)
        except asyncio.TimeoutError:
            # Best effort: frees the Serve slot. The replica's executor thread
            # still runs the inference to completion (threads can't be killed).
            cancel = getattr(response, "cancel", None)
            if cancel is not None:
                try:
                    cancel()
                except Exception:
                    pass
            raise
