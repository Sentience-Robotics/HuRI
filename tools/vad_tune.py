#!/usr/bin/env python
"""Find the VAD settings that actually close a turn on *your* microphone.

Why this exists
---------------
``MIC`` (src/modules/speech_to_text/microphone_vad.py) decides a turn is over
when WebRTC's VAD has reported ``silence_duration`` seconds of non-speech in a
row. Everything downstream is gated on that one decision: no end marker means
STT never emits ``Transcript(end=True)``, TAG never flushes its sentence, QAG
never builds a RAGQuestion — and the browser never sees a "question" message.
The prompt is simply never sent, with nothing in the UI to say why.

Whether that decision is right depends entirely on the signal your capture path
produces: room noise, mic gain, and (in the browser) Chrome's echo cancellation
and noise suppression all move the line between "speech" and "silence". A
setting that works on one machine drops turns on another, which is why this is
a measurement rather than a default.

Getting a recording
-------------------
Tune on the audio HuRI actually scored, not on a fresh capture — the browser
path resamples and denoises, so a local recording is a different signal, and on
WSL2 there is no capture device at all. Set ``HURI_MIC_DUMP_DIR`` when starting
HuRI and MIC writes every frame it is handed to a WAV::

    HURI_MIC_DUMP_DIR=/tmp/huri-mic serve run config/huri.yaml

Then talk to the site as usual: a few sentences with a clear pause between
them, plus a few seconds of just-sitting-there at the start so the tool can
measure your noise floor.

Usage
-----
::

    tools/vad_tune.py level /tmp/huri-mic/mic-*.wav
    tools/vad_tune.py tune  /tmp/huri-mic/mic-*.wav --utterances 3
    tools/vad_tune.py trace /tmp/huri-mic/mic-*.wav -a 2 -s 1.0

``level`` first: if the recording is clipped, near-silent, or has no usable gap
between speech and background, no VAD setting will save it and the tuner will
only tell you which flavour of wrong is least wrong.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Importing the shipped MIC (rather than reimplementing its state machine) is
# the whole point: a tuner that drifts from production tells you about a VAD
# that isn't the one running. The dump env var is cleared first so replaying a
# capture doesn't write another capture.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("HURI_MIC_DUMP_DIR", None)

from src.core.events import BytesEvent  # noqa: E402
from src.modules.speech_to_text.microphone_vad import MIC  # noqa: E402

# MIC logs a line per speech onset and per closed turn, which is what you want
# in a live session and useless here: one sweep replays the recording ~28
# times. Has to come after the imports — importing src.core.module pulls in
# ray, which installs its own handler on this logger and resets the level.
# Errors still come through: a frame-size mismatch is worth seeing.
logging.getLogger("ray.serve").setLevel(logging.ERROR)

SAMPLE_RATE = 16000

# The browser client emits fixed 480-sample (30 ms) frames — see
# HuRI_ATP_testsite/frontend/src/audio/microphone.js, where the size is
# hardcoded because WebRTC VAD only accepts 10/20/30 ms. So 0.03 is the only
# block_duration the website can actually feed; the others are offered behind
# --all-blocks for the CLI client, whose frame size is configurable.
WEB_BLOCK_DURATION = 0.030
ALL_BLOCK_DURATIONS = (0.010, 0.020, 0.030)

AGGRESSIVENESS_GRID = (0, 1, 2, 3)
SILENCE_GRID = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)


# ---------------------------------------------------------------------------
# Recording I/O
# ---------------------------------------------------------------------------


def load_wav(path: Path) -> np.ndarray:
    """Read a mono 16 kHz 16-bit WAV into an int16 array.

    Refuses anything else rather than resampling: the point is to score the
    exact samples MIC was handed, and a conversion here would tune the VAD on a
    signal that never existed.
    """
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(
                f"{path}: expected mono 16-bit PCM, got "
                f"{w.getnchannels()} channel(s) / {w.getsampwidth() * 8}-bit"
            )
        if w.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: expected {SAMPLE_RATE} Hz, got {w.getframerate()} Hz. "
                "HuRI's mic path is 16 kHz end to end; a file at another rate "
                "did not come from it."
            )
        frames = w.getnframes()
        if frames == 0:
            raise SystemExit(
                f"{path} is empty. If HuRI is still running, talk to it first — "
                "the dump is written as frames arrive."
            )
        return np.frombuffer(w.readframes(frames), dtype=np.int16)


def to_blocks(samples: np.ndarray, block_duration: float) -> List[bytes]:
    """Slice int16 samples into whole VAD-sized frames, dropping the remainder."""
    block_size = int(block_duration * SAMPLE_RATE)
    count = len(samples) // block_size
    return [
        samples[i * block_size : (i + 1) * block_size].tobytes() for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Signal health
# ---------------------------------------------------------------------------


@dataclass
class LevelReport:
    duration: float
    peak_dbfs: float
    noise_dbfs: float
    speech_dbfs: float
    dc_offset: float
    clipped_frames: int
    total_frames: int

    @property
    def headroom(self) -> float:
        """dB between the background floor and the loud parts.

        This — not the absolute level — is what a VAD has to work with. Below
        ~15 dB, "speech" and "room" overlap and every aggressiveness setting is
        a bad trade.
        """
        return self.speech_dbfs - self.noise_dbfs


def frame_dbfs(samples: np.ndarray, block_duration: float) -> np.ndarray:
    """Per-frame RMS in dBFS (-100 for a digitally silent frame)."""
    block_size = int(block_duration * SAMPLE_RATE)
    count = len(samples) // block_size
    blocks = samples[: count * block_size].reshape(count, block_size).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(blocks / 32768.0), axis=1))
    db: np.ndarray = 20.0 * np.log10(np.maximum(rms, 1e-5))
    return db


def analyse_level(samples: np.ndarray, block_duration: float) -> LevelReport:
    db = frame_dbfs(samples, block_duration)
    peak = float(np.max(np.abs(samples))) / 32768.0
    return LevelReport(
        duration=len(samples) / SAMPLE_RATE,
        peak_dbfs=20.0 * math.log10(max(peak, 1e-5)),
        # 10th/90th percentile rather than min/max so one door slam or one
        # dropout doesn't define the whole recording.
        noise_dbfs=float(np.percentile(db, 10)),
        speech_dbfs=float(np.percentile(db, 90)),
        dc_offset=float(np.mean(samples)) / 32768.0,
        clipped_frames=int(np.sum(np.abs(samples) >= 32700)),
        total_frames=len(samples),
    )


def print_level(report: LevelReport) -> None:
    print(f"  duration        {report.duration:7.2f} s")
    print(f"  peak            {report.peak_dbfs:7.1f} dBFS")
    print(f"  background      {report.noise_dbfs:7.1f} dBFS  (10th pct frame RMS)")
    print(f"  speech          {report.speech_dbfs:7.1f} dBFS  (90th pct frame RMS)")
    print(f"  headroom        {report.headroom:7.1f} dB    (speech - background)")
    print(f"  DC offset       {report.dc_offset:7.4f}")
    clipped_pct = 100.0 * report.clipped_frames / max(report.total_frames, 1)
    print(f"  clipped samples {clipped_pct:7.2f} %")

    # Speech normally peaks around -12 dBFS. Well below that and the VAD is
    # scoring the bottom few bits of the converter, which is a gain problem no
    # amount of tuning fixes.
    warnings: List[str] = []
    if report.peak_dbfs < -25:
        warnings.append(
            "Very quiet capture. Raise the input gain in your OS mixer — at "
            "this level the VAD is scoring mostly quantisation noise, and no "
            "setting below will fix that."
        )
    if clipped_pct > 0.5:
        warnings.append(
            "Clipping. Lower the input gain; a clipped waveform is broadband "
            "and reads as speech, so turns will refuse to close."
        )
    if abs(report.dc_offset) > 0.01:
        warnings.append("Significant DC offset — it inflates every frame's RMS.")
    if report.headroom < 15:
        warnings.append(
            f"Only {report.headroom:.0f} dB between speech and background. "
            "WebRTC's VAD needs roughly 15-20 dB to separate them; expect to "
            "trade dropped words against turns that never end. Move closer to "
            "the mic or cut the room noise."
        )

    print()
    for warning in warnings:
        print(f"  ! {warning}")
    if not warnings:
        print("  Healthy capture: good level, no clipping, clear speech/room gap.")


# ---------------------------------------------------------------------------
# Reference segmentation
# ---------------------------------------------------------------------------


@dataclass
class Utterance:
    """A run of speech in the reference segmentation, in frame indices."""

    start: int
    end: int  # exclusive

    def seconds(self, block_duration: float) -> Tuple[float, float]:
        return self.start * block_duration, self.end * block_duration


def reference_utterances(
    samples: np.ndarray,
    block_duration: float,
    pause: float,
    min_utterance: float,
) -> Tuple[List[Utterance], np.ndarray]:
    """Segment the recording by energy, independently of WebRTC's VAD.

    Scoring a VAD against itself proves nothing, so ground truth comes from a
    plain adaptive-threshold detector: the level analysis already tells us
    where the background sits, and speech is what rises well clear of it.
    Hysteresis (a lower exit threshold than entry) stops a single dipping frame
    from splitting a word.

    :param pause: gaps shorter than this are breaths inside one sentence, not
        turn boundaries, and get merged.
    :param min_utterance: runs shorter than this are lip smacks and keystrokes.
    :returns: the utterances, and the per-frame speech mask they came from.
    """
    db = frame_dbfs(samples, block_duration)
    noise = float(np.percentile(db, 10))
    loud = float(np.percentile(db, 95))
    # Sit the entry threshold a third of the way up from the floor to the loud
    # parts, but never so close to the floor that background crosses it.
    enter = max(noise + 0.33 * (loud - noise), noise + 8.0)
    exit_ = enter - 3.0

    mask = np.zeros(len(db), dtype=bool)
    active = False
    for i, value in enumerate(db):
        if active:
            active = value > exit_
        else:
            active = value > enter
        mask[i] = active

    runs = _runs(mask)
    merged = _merge_runs(runs, int(round(pause / block_duration)))
    min_frames = int(round(min_utterance / block_duration))
    return [u for u in merged if u.end - u.start >= min_frames], mask


def _runs(mask: np.ndarray) -> List[Utterance]:
    out: List[Utterance] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append(Utterance(start, i))
            start = None
    if start is not None:
        out.append(Utterance(start, len(mask)))
    return out


def _merge_runs(runs: Sequence[Utterance], max_gap: int) -> List[Utterance]:
    out: List[Utterance] = []
    for run in runs:
        if out and run.start - out[-1].end <= max_gap:
            out[-1] = Utterance(out[-1].start, run.end)
        else:
            out.append(Utterance(run.start, run.end))
    return out


def reconcile(
    utterances: List[Utterance],
    mask: np.ndarray,
    expected: Optional[int],
) -> Tuple[List[Utterance], np.ndarray, Optional[str]]:
    """Bring the energy segmentation in line with what you say you said.

    You know how many times you spoke; the threshold detector is guessing. When
    it guesses high it has latched onto a chair creak or a keyboard press, and
    leaving that in the reference would mark every candidate config as having
    "missed a turn" — the one metric that decides the ranking. So the extra
    runs are dropped, longest-first being kept, and their frames are demoted to
    reference-silence, where a VAD firing on them correctly shows up as noise
    leak instead.

    Guessing low is not fixable here: two sentences you separated by a short
    breath really did merge into one run. Say so and let the caller lower
    --pause.
    """
    if not expected or expected == len(utterances):
        return utterances, mask, None

    if expected > len(utterances):
        return (
            utterances,
            mask,
            f"energy segmentation found only {len(utterances)} — two of your "
            "sentences probably ran together; lower --pause if the gap between "
            "them was short",
        )

    ranked = sorted(utterances, key=lambda u: u.end - u.start, reverse=True)
    keep = set(id(u) for u in ranked[:expected])
    dropped = [u for u in utterances if id(u) not in keep]

    mask = mask.copy()
    for run in dropped:
        mask[run.start : run.end] = False

    return (
        [u for u in utterances if id(u) in keep],
        mask,
        f"dropped {len(dropped)} short run(s) as noise, not speech",
    )


# ---------------------------------------------------------------------------
# Replaying MIC
# ---------------------------------------------------------------------------


@dataclass
class Replay:
    """What the real MIC state machine did over a recording."""

    speech: np.ndarray  # per-frame: did MIC emit a Voice(data)?
    ends: List[int] = field(default_factory=list)  # frames that emitted Voice(None)


async def _replay(
    blocks: Sequence[bytes],
    aggressiveness: int,
    silence_duration: float,
    block_duration: float,
) -> Replay:
    mic = MIC(
        vad_agressiveness=aggressiveness,
        silence_duration=silence_duration,
        block_duration=block_duration,
    )
    speech = np.zeros(len(blocks), dtype=bool)
    ends: List[int] = []
    for i, block in enumerate(blocks):
        voice = await mic.process(BytesEvent(data=block))
        if voice is None:
            continue
        if voice.data is None:
            ends.append(i)
        else:
            speech[i] = True
    return Replay(speech=speech, ends=ends)


def replay(
    blocks: Sequence[bytes],
    aggressiveness: int,
    silence_duration: float,
    block_duration: float,
) -> Replay:
    return asyncio.run(
        _replay(blocks, aggressiveness, silence_duration, block_duration)
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class Result:
    aggressiveness: int
    silence_duration: float
    block_duration: float

    turns: int
    expected: int
    missed: int  # utterances that never produced an end marker
    splits: int  # end markers that fired in the middle of an utterance
    coverage: float  # share of reference speech frames MIC forwarded
    leak: float  # share of reference silence frames MIC called speech
    latency: float  # mean seconds from end of speech to the end marker

    @property
    def turn_error(self) -> int:
        return abs(self.turns - self.expected)

    @property
    def usable(self) -> bool:
        """Would every sentence you spoke have reached RAG?"""
        return self.missed == 0 and self.splits == 0 and self.turns == self.expected

    def sort_key(self) -> Tuple:
        """Rank by what breaks the feature, worst failure first.

        A missed turn is the reported bug (nothing is ever sent), so it
        dominates. A split turn sends half a sentence, which is wrong but
        visible. Only once both are zero do coverage (words reaching Whisper)
        and latency (how long you wait after speaking) decide.
        """
        return (
            self.missed,
            self.turn_error,
            self.splits,
            -round(self.coverage, 3),
            round(self.latency, 2),
            round(self.leak, 3),
        )


def score(
    rep: Replay,
    utterances: Sequence[Utterance],
    reference_mask: np.ndarray,
    block_duration: float,
    aggressiveness: int,
    silence_duration: float,
) -> Result:
    """Compare one config's replay against the reconciled reference."""
    ends = list(rep.ends)
    missed = 0
    splits = 0
    latencies: List[float] = []

    for index, utterance in enumerate(utterances):
        # The marker for this utterance is the first one at or after its last
        # speech frame, but before the next utterance starts talking.
        limit = utterances[index + 1].start if index + 1 < len(utterances) else None
        candidates = [
            e for e in ends if e >= utterance.end - 1 and (limit is None or e < limit)
        ]
        if candidates:
            latencies.append((candidates[0] - (utterance.end - 1)) * block_duration)
        else:
            missed += 1

        # A marker strictly inside an utterance means silence_duration is
        # shorter than a pause you take mid-sentence: the turn is cut and the
        # second half opens a new one.
        splits += sum(1 for e in ends if utterance.start < e < utterance.end - 1)

    speech_frames = int(np.sum(reference_mask))
    silence_frames = int(np.sum(~reference_mask))
    coverage = (
        float(np.sum(rep.speech & reference_mask)) / speech_frames
        if speech_frames
        else 0.0
    )
    leak = (
        float(np.sum(rep.speech & ~reference_mask)) / silence_frames
        if silence_frames
        else 0.0
    )

    return Result(
        aggressiveness=aggressiveness,
        silence_duration=silence_duration,
        block_duration=block_duration,
        turns=len(ends),
        expected=len(utterances),
        missed=missed,
        splits=splits,
        coverage=coverage,
        leak=leak,
        latency=float(np.mean(latencies)) if latencies else float("inf"),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_level(args: argparse.Namespace) -> int:
    samples = load_wav(args.wav)
    print(f"\nSignal health — {args.wav}\n")
    print_level(analyse_level(samples, WEB_BLOCK_DURATION))
    utterances, _ = reference_utterances(
        samples, WEB_BLOCK_DURATION, args.pause, args.min_utterance
    )
    print()
    print(f"  reference segmentation: {len(utterances)} utterance(s)")
    for i, u in enumerate(utterances, 1):
        start, end = u.seconds(WEB_BLOCK_DURATION)
        print(f"    {i:2}. {start:6.2f}s -> {end:6.2f}s  ({end - start:.2f}s)")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    samples = load_wav(args.wav)
    report = analyse_level(samples, WEB_BLOCK_DURATION)

    print(f"\nSignal health — {args.wav}\n")
    print_level(report)

    utterances, mask = reference_utterances(
        samples, WEB_BLOCK_DURATION, args.pause, args.min_utterance
    )
    if not utterances:
        print(
            "\nNo speech found. Either the recording really is silence, or the "
            "level is too low to segment — check the numbers above before "
            "touching any VAD setting."
        )
        return 1

    utterances, mask, note = reconcile(utterances, mask, args.utterances)
    print(f"\nReference: {len(utterances)} utterance(s)", end="")
    print(f" — {note}." if note else " detected by energy.")
    for i, u in enumerate(utterances, 1):
        start, end = u.seconds(WEB_BLOCK_DURATION)
        print(f"    {i:2}. {start:6.2f}s -> {end:6.2f}s  ({end - start:.2f}s)")

    results: List[Result] = []
    for block_duration in args.block_durations:
        blocks = to_blocks(samples, block_duration)
        # Re-segment per block size so reference frame indices line up with the
        # replay's, then reconcile again against the same expected count.
        ref_utterances, ref_mask = reference_utterances(
            samples, block_duration, args.pause, args.min_utterance
        )
        ref_utterances, ref_mask, _ = reconcile(
            ref_utterances, ref_mask, args.utterances
        )
        for aggressiveness in AGGRESSIVENESS_GRID:
            for silence in SILENCE_GRID:
                rep = replay(blocks, aggressiveness, silence, block_duration)
                results.append(
                    score(
                        rep,
                        ref_utterances,
                        ref_mask,
                        block_duration,
                        aggressiveness,
                        silence,
                    )
                )

    results.sort(key=Result.sort_key)
    _print_table(results[: args.top])

    best = results[0]
    print()
    if not best.usable:
        print(
            "No setting in the sweep cleanly closes every turn on this "
            "recording. The best compromise is shown first; read the signal "
            "health above — this is usually gain or room noise, not config."
        )
    _print_config(best)
    return 0 if best.usable else 2


def _print_table(results: Iterable[Result]) -> None:
    print()
    print(
        f"  {'aggr':>4} {'silence':>8} {'block':>6} "
        f"{'turns':>6} {'missed':>7} {'split':>6} "
        f"{'covered':>8} {'noise':>7} {'delay':>7}"
    )
    print("  " + "-" * 70)
    for r in results:
        latency = "-" if math.isinf(r.latency) else f"{r.latency:.2f}s"
        flag = "  <- ok" if r.usable else ""
        print(
            f"  {r.aggressiveness:>4} {r.silence_duration:>7.1f}s "
            f"{r.block_duration * 1000:>5.0f}ms "
            f"{r.turns:>3}/{r.expected:<2} {r.missed:>7} {r.splits:>6} "
            f"{r.coverage * 100:>7.0f}% {r.leak * 100:>6.0f}% {latency:>7}{flag}"
        )
    print()
    print("  covered = share of your speech that reached Whisper")
    print("  noise   = share of the quiet parts the VAD mistook for speech")
    print("  delay   = wait between you finishing and the turn closing")


def _print_config(best: Result) -> None:
    block = best.block_duration
    print("Paste into the preset (HuRI_ATP_testsite/presets/<F*>/<name>.json),")
    print("or into the Event Configuration modal's args box:\n")
    print(
        '  "mic": {{ "name": "mic", "args": {{ "vad_agressiveness": {a}, '
        '"silence_duration": {s}, "block_duration": {b} }} }},'.format(
            a=best.aggressiveness, s=best.silence_duration, b=block
        )
    )
    print(f'  "stt": {{ "name": "stt", "args": {{ "block_duration": {block} }} }},')
    print(f'  "emo": {{ "name": "emo", "args": {{ "block_duration": {block} }} }}')
    print()
    print(
        "mic/stt/emo must all carry the same block_duration — STT sizes its "
        "transcription window in frames, so a mismatch silently changes how "
        "much audio a window holds."
    )
    if abs(block - WEB_BLOCK_DURATION) > 1e-9:
        print()
        print(
            f"! {block * 1000:.0f} ms frames need a frontend change too: the "
            "browser client hardcodes 480-sample/30 ms frames (FRAME_SAMPLES in "
            "frontend/src/audio/microphone.js). Without it MIC drops every "
            "frame. Prefer a 30 ms row unless you are tuning the CLI client."
        )


def cmd_trace(args: argparse.Namespace) -> int:
    samples = load_wav(args.wav)
    block = args.block
    blocks = to_blocks(samples, block)
    rep = replay(blocks, args.aggressiveness, args.silence, block)
    db = frame_dbfs(samples, block)

    print(
        f"\nTimeline — aggressiveness {args.aggressiveness}, "
        f"silence_duration {args.silence}s, block {block * 1000:.0f}ms\n"
    )
    print(f"  {'time':>8} {'dBFS':>7}  VAD")
    ends = set(rep.ends)
    was_speech = False
    for i in range(len(blocks)):
        speech = bool(rep.speech[i])
        end = i in ends
        # Only print transitions and end markers: a 30-second capture is 1000
        # frames and a wall of identical lines hides the three that matter.
        if speech == was_speech and not end:
            continue
        marker = "END OF TURN" if end else ("speech" if speech else "silence")
        print(f"  {i * block:7.2f}s {db[i]:7.1f}  {marker}")
        was_speech = speech
    print(f"\n  {len(rep.ends)} turn(s) closed.")
    if not rep.ends:
        print(
            "  Nothing closed a turn: with these settings the prompt is never "
            "sent. Run `tune` on this file."
        )
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Capture locally. Only useful where a capture device exists."""
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("sounddevice is not installed in this environment.")

    if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
        raise SystemExit(
            "No input device. On WSL2 there is none — record through the "
            "browser instead: start HuRI with HURI_MIC_DUMP_DIR=<dir>, talk to "
            "the site, then tune the WAV it writes. That is the better signal "
            "anyway: it is what MIC actually scored, after the browser's echo "
            "cancellation and noise suppression."
        )

    print(f"Recording {args.seconds}s at {SAMPLE_RATE} Hz — speak, pause, speak...")
    audio = sd.rec(
        int(args.seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    print(f"Wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _block_durations(all_blocks: bool) -> Tuple[float, ...]:
    return ALL_BLOCK_DURATIONS if all_blocks else (WEB_BLOCK_DURATION,)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_segmentation_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pause",
            type=float,
            default=0.35,
            help="gaps shorter than this are treated as pauses inside one "
            "sentence, not turn boundaries (default: %(default)s s)",
        )
        p.add_argument(
            "--min-utterance",
            type=float,
            default=0.4,
            help="ignore speech runs shorter than this (default: %(default)s s)",
        )

    p_level = sub.add_parser("level", help="signal health of a recording")
    p_level.add_argument("wav", type=Path)
    add_segmentation_args(p_level)
    p_level.set_defaults(func=cmd_level)

    p_tune = sub.add_parser("tune", help="sweep VAD settings and rank them")
    p_tune.add_argument("wav", type=Path)
    p_tune.add_argument(
        "--utterances",
        type=int,
        help="how many separate things you said (defaults to what the energy "
        "segmentation finds)",
    )
    p_tune.add_argument(
        "--all-blocks",
        action="store_true",
        help="also sweep 10/20 ms frames (needs a frontend change to use)",
    )
    p_tune.add_argument("--top", type=int, default=12, help="rows to show")
    add_segmentation_args(p_tune)
    p_tune.set_defaults(func=cmd_tune)

    p_trace = sub.add_parser("trace", help="frame-by-frame timeline for one config")
    p_trace.add_argument("wav", type=Path)
    p_trace.add_argument(
        "-a", "--aggressiveness", type=int, default=2, choices=(0, 1, 2, 3)
    )
    p_trace.add_argument("-s", "--silence", type=float, default=1.0)
    p_trace.add_argument(
        "-b",
        "--block",
        type=float,
        default=WEB_BLOCK_DURATION,
        choices=ALL_BLOCK_DURATIONS,
    )
    p_trace.set_defaults(func=cmd_trace)

    p_record = sub.add_parser("record", help="capture locally (needs an input device)")
    p_record.add_argument("out", type=Path)
    p_record.add_argument("--seconds", type=float, default=20.0)
    p_record.set_defaults(func=cmd_record)

    args = parser.parse_args(argv)
    if args.command == "tune":
        args.block_durations = _block_durations(args.all_blocks)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
