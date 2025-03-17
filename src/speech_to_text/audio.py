import pyaudio
import numpy as np
from scipy.signal import butter, filtfilt
import wave

import pyaudio
import time
from scipy.signal import butter, filtfilt

from pynput import keyboard
import os
import sys

import termios
import atexit


def enable_echo(fd, enabled):
    (iflag, oflag, cflag, lflag, ispeed, ospeed, cc) = termios.tcgetattr(fd)

    if enabled:
        lflag |= termios.ECHO
    else:
        lflag &= ~termios.ECHO

    new_attr = [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
    termios.tcsetattr(fd, termios.TCSANOW, new_attr)


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def highpass_filter(data, b, a):
    y = filtfilt(b, a, data)
    return y


def record_and_process_audio(
    silence_threshold=500,
    silence_duration=5,
    rate=44100,
    chunk=1024,
    cutoff_frequency=300,
):

    format = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk
    )

    print("Press Enter to start recording.", flush=True)
    frames = []
    silence_counter = 0

    b, a = butter_highpass(cutoff_frequency, rate)
    recording = False

    def on_press(key):
        nonlocal recording
        if not recording and key == keyboard.Key.enter:
            recording = True
            enable_echo(sys.stdin.fileno(), False)
            print("Recording started.", flush=True)

    def on_release(key):
        nonlocal recording
        if key == keyboard.Key.enter:
            recording = False
            enable_echo(sys.stdin.fileno(), True)
            return False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    atexit.register(enable_echo, sys.stdin.fileno(), True)
    listener.start()

    while not recording:
        time.sleep(0.01)

    while recording:
        data = stream.read(chunk)
        audio_data = np.frombuffer(data, dtype=np.int16)

        filtered_data = highpass_filter(audio_data, b, a)
        filtered_data = filtered_data.astype(np.int16).tobytes()
        frames.append(filtered_data)

    print("Recording end.", flush=True)
    listener.join()
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return frames


def save_audio(frames, filename="test_audio/processed_audio.wav", rate=44100):
    """Saves the processed audio frames to a file."""
    format = pyaudio.paInt16
    channels = 1

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    print(f"Processed audio saved as {filename}")


def speech_to_wav(filename: None | str, rate=44100):
    frames = record_and_process_audio(rate=rate)
    if filename != None:
        save_audio(frames, filename, rate)

    return frames
