import json
import logging
import math
import os
import struct
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
import webrtcvad

from src.core.events import BytesEvent
from src.core.module import Module

from .events import Voice

logger = logging.getLogger("ray.serve")


def _to_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def _dbfs(pcm: np.ndarray) -> float:
    if len(pcm) == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) / 32768.0
    return 20 * math.log10(rms + 1e-9)


class _FrameDump:
    """Capture of every frame MIC scores, for offline VAD tuning.

    Enabled by ``HURI_MIC_DUMP_DIR`` (or MIC's ``dump_dir``). Writes a mono
    int16 WAV plus a ``.jsonl`` sidecar with MIC's per-frame decision, which is
    what ``tools/vad_tune.py`` replays. The WAV header is re-patched about once
    per second so the file stays readable even if the replica dies.
    """

    def __init__(self, directory: Optional[str], sample_rate: int):
        self._wav = None
        self._marks = None
        self.path: Optional[str] = None
        if not directory:
            return
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            directory, f"mic_{stamp}_{os.getpid()}_{id(self):x}.wav"
        )
        self._sample_rate = sample_rate
        self._data_bytes = 0
        self._since_patch = 0
        self._wav = open(self.path, "wb")
        self._marks = open(self.path + ".jsonl", "w")
        self._write_header()
        logger.info("[MIC] dumping frames to %s", self.path)

    def _write_header(self) -> None:
        sr = self._sample_rate
        self._wav.seek(0)
        self._wav.write(
            b"RIFF"
            + struct.pack("<I", 36 + self._data_bytes)
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data"
            + struct.pack("<I", self._data_bytes)
        )
        self._wav.seek(0, os.SEEK_END)

    def write(self, raw: bytes, speech: bool, in_utterance: bool) -> None:
        if self._wav is None:
            return
        self._wav.write(raw)
        self._data_bytes += len(raw)
        self._since_patch += len(raw)
        self._marks.write(
            json.dumps(
                {
                    "t": round(self._data_bytes / 2 / self._sample_rate, 3),
                    "speech": speech,
                    "in": in_utterance,
                }
            )
            + "\n"
        )
        if self._since_patch >= self._sample_rate * 2:  # ~1 s of int16 audio
            self._write_header()
            self._wav.flush()
            self._marks.flush()
            self._since_patch = 0

    def close(self) -> None:
        if self._wav is None:
            return
        self._write_header()
        self._wav.close()
        self._marks.close()
        self._wav = None
        self._marks = None


class MIC(Module):
    """MIC Module

    Detect voice and silence using WebRTC VAD, and cut the stream into
    utterances ("turns").

    input: audio.in,
    output: voice

    Emits, per turn: one ``Voice`` holding the pre-roll at speech onset, then
    one ``Voice`` per frame — speech or not, so Whisper hears the natural pauses
    between words — and finally exactly one ``Voice(None)`` when the turn ends:
    after ``silence_duration`` of non-speech, when the utterance hits
    ``max_utterance_duration`` (a noisy room must not be able to hold a turn
    open forever: everything downstream waits on that single marker), or when
    the client sends an EMPTY ``audio.in`` frame — its way of saying "I closed
    the microphone", see :meth:`flush`. Frames outside a turn are dropped.

    :vad_agressiveness: from 0 (low) to 3 (high, can distord audio).
    :silence_duration: how many seconds will a no voice be considered a silence.
    :sample_rate: size of received chunk of audio. Usually 8000, 16000 or 48000.
    :block_duration: size of received chunk of audio (in s).
        Can only be 0.010, 0.020 and 0.030.
    :pre_roll_duration: audio kept from before the onset so word starts are not
        clipped (in s).
    :speech_debounce_frames: consecutive VAD-positive frames needed to open a
        turn, and to reset the silence countdown inside one — a lone noise
        frame no longer extends the turn.
    :max_utterance_duration: force the turn closed after this long (in s).
    :min_rms_dbfs: optional level floor; a VAD-positive frame quieter than this
        counts as silence (faint background chatter). None disables it.
    :dump_dir: write every frame to a WAV for tools/vad_tune.py. Defaults to
        the HURI_MIC_DUMP_DIR env var.
    """

    # Inbound microphone frames travel on their own topic so the TTS-output
    # "audio" topic (consumed by Gesture and the client Sender) never collides
    # with mic input — otherwise raw mic bytes get echoed back to the client.
    input_type = "audio.in"
    output_type = "voice"

    def __init__(
        self,
        vad_agressiveness: int = 3,
        silence_duration: float = 3,  # s
        sample_rate: int = 16000,
        block_duration: float = 0.020,  # s
        pre_roll_duration: float = 0.3,  # s
        speech_debounce_frames: int = 2,
        max_utterance_duration: float = 20.0,  # s
        min_rms_dbfs: Optional[float] = None,
        dump_dir: Optional[str] = None,
    ):
        super().__init__()

        if block_duration not in [0.010, 0.020, 0.030]:
            raise RuntimeError("block duration must be 0.010, 0.020 or 0.030 s")

        self.sample_rate: int = sample_rate
        self.block_size: int = int(block_duration * sample_rate)

        self.silence_samples_max: int = int(silence_duration * sample_rate)
        self.max_utterance_samples: int = int(max_utterance_duration * sample_rate)
        self.debounce: int = max(1, int(speech_debounce_frames))
        self.min_rms_dbfs = min_rms_dbfs

        self.vad = webrtcvad.Vad(vad_agressiveness)

        n_pre = max(self.debounce, int(round(pre_roll_duration / block_duration)))
        self.pre_roll: Deque[np.ndarray] = deque(maxlen=n_pre)

        self.in_utterance: bool = False
        self.speech_run: int = 0  # consecutive VAD-positive frames
        self.silence_samples: int = 0  # trailing non-speech inside the turn
        self.utterance_samples: int = 0
        self.utterance_frames: int = 0
        self.turn: int = 0
        self._t_onset: float = 0.0
        self._bad_frame_logged: bool = False

        self._dump = _FrameDump(
            dump_dir or os.environ.get("HURI_MIC_DUMP_DIR"), sample_rate
        )

    def _is_speech(self, raw: bytes, pcm: np.ndarray) -> bool:
        try:
            speech = bool(self.vad.is_speech(raw, self.sample_rate))
        except Exception:
            # Wrong frame length. Treat as silence instead of raising — the bus
            # would swallow the exception per frame and nothing would tell you.
            if not self._bad_frame_logged:
                logger.warning(
                    "[MIC] undecodable frame (%d bytes): expected 10/20/30 ms "
                    "of int16 mono @ %d Hz",
                    len(raw),
                    self.sample_rate,
                )
                self._bad_frame_logged = True
            return False
        if speech and self.min_rms_dbfs is not None and _dbfs(pcm) < self.min_rms_dbfs:
            return False
        return speech

    async def process(self, data: BytesEvent) -> Optional[Voice]:
        raw = data.data
        if not raw:
            return self.flush()
        pcm = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype=np.int16)
        speech = self._is_speech(raw, pcm)
        out = self._step(pcm, speech)
        # "in" records the state AFTER this frame: whether it was forwarded as
        # part of a turn (the closing silence frame is not).
        self._dump.write(raw, speech, self.in_utterance)
        return out

    def _step(self, pcm: np.ndarray, speech: bool) -> Optional[Voice]:
        if not self.in_utterance:
            self.pre_roll.append(pcm)
            self.speech_run = self.speech_run + 1 if speech else 0
            if self.speech_run < self.debounce:
                return None
            return self._open()

        # Inside a turn: forward every frame, count trailing silence in real
        # samples (not the configured block size — clients may differ).
        self.utterance_samples += len(pcm)
        self.utterance_frames += 1
        if speech:
            self.speech_run += 1
            if self.speech_run >= self.debounce:
                self.silence_samples = 0
        else:
            self.speech_run = 0
            self.silence_samples += len(pcm)

        if self.silence_samples >= self.silence_samples_max:
            return self._close("silence")
        if self.utterance_samples >= self.max_utterance_samples:
            marker = self._close("max_duration")
            # The user is still talking: seed the next turn with this frame so
            # it opens on the very next speech frame.
            self.pre_roll.append(pcm)
            self.speech_run = 1 if speech else 0
            return marker
        return Voice(_to_float(pcm))

    def _open(self) -> Voice:
        self.in_utterance = True
        self.turn += 1
        self.silence_samples = 0
        self._t_onset = time.monotonic()
        onset = np.concatenate(list(self.pre_roll))
        self.pre_roll.clear()
        self.utterance_samples = len(onset)
        self.utterance_frames = 1
        logger.info(
            "[MIC] turn %d onset (pre-roll %d ms)",
            self.turn,
            1000 * len(onset) // self.sample_rate,
        )
        return Voice(_to_float(onset))

    def flush(self) -> Optional[Voice]:
        """End the turn NOW, without waiting for the silence countdown.

        Triggered by an empty ``audio.in`` frame, which a client sends when it
        closes its microphone (the browser's mic button, see the web
        interface). No more frames are coming after that, so the
        ``silence_duration`` of non-speech that normally closes a turn would
        never be observed: the turn would stay open, STT would never run its
        final pass on what was said, and the first frames of the next
        recording — possibly minutes later — would be glued onto it as a
        continuation. The pre-roll is dropped for the same reason.

        Returns the end marker, or None when no turn was open (nothing to
        flush; the marker is only ever emitted once per turn).
        """
        if not self.in_utterance:
            self.pre_roll.clear()
            self.speech_run = 0
            logger.info("[MIC] flush outside a turn, nothing to close")
            return None
        return self._close("flush")

    def _close(self, reason: str) -> Voice:
        log = logger.warning if reason == "max_duration" else logger.info
        log(
            "[MIC] turn %d close reason=%s audio=%.2fs frames=%d wall=%.2fs",
            self.turn,
            reason,
            self.utterance_samples / self.sample_rate,
            self.utterance_frames,
            time.monotonic() - self._t_onset,
        )
        self.in_utterance = False
        self.speech_run = 0
        self.silence_samples = 0
        self.utterance_samples = 0
        self.utterance_frames = 0
        # Already forwarded — otherwise the next onset re-emits this tail.
        self.pre_roll.clear()
        return Voice(None)  # exactly once per turn, by construction

    async def finalize(self) -> None:
        self._dump.close()
