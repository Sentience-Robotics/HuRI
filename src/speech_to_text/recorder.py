import speech_recognition as sr
import wave
import pyaudio
import os
from speech_to_text.audio import speech_to_wav

# def record_audio(duration=5, filename="test_audio/my_audio.wav", format=pyaudio.paInt16, channels=1, rate=44100):
#     """Records audio for a given duration and saves it to a file."""
#     chunk = 1024  # Record in chunks of 1024 samples

#     audio = pyaudio.PyAudio()

#     try:
#         stream = audio.open(format=format, channels=channels,
#                             rate=rate, input=True, frames_per_buffer=chunk)
#         print("Recording...")
#         frames = []

#         for _ in range(0, int(rate / chunk * duration)):
#             data = stream.read(chunk)
#             frames.append(data)

#         print("Recording finished")
#         stream.stop_stream()
#         stream.close()

#         # with wave.open(filename, 'wb') as wf:
#         #     wf.setnchannels(channels)
#         #     wf.setsampwidth(audio.get_sample_size(format))
#         #     wf.setframerate(rate)
#         #     wf.writeframes(b''.join(frames))

#         # print(f"Audio saved as {filename}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

#     finally:
#         # Terminate the PyAudio object
#         audio.terminate()
#         return frames

# import wave
# import pyaudio

# def record_audio(duration=5, filename="test_audio/my_audio.wav"):
#     """Records audio for a given duration and saves it to a file."""
#     # Set parameters for audio recording
#     format = pyaudio.paInt16  # 16-bit resolution
#     channels = 1  # Mono sound
#     rate = 44100  # Sample rate
#     chunk = 1024  # Record in chunks of 1024 samples

#     # Initialize the PyAudio object
#     audio = pyaudio.PyAudio()

#     # Start recording
#     stream = audio.open(format=format, channels=channels,
#                         rate=rate, input=True, frames_per_buffer=chunk)
#     print("Recording...")
#     frames = []

#     for _ in range(0, int(rate / chunk * duration)):
#         data = stream.read(chunk)
#         frames.append(data)

#     # Stop and close the stream
#     print("Recording finished")
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     # Save recorded audio to file
#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(channels)
#         wf.setsampwidth(audio.get_sample_size(format))
#         wf.setframerate(rate)
#         wf.writeframes(b''.join(frames))

#     print(f"Audio saved as {filename}")


def listen_for_keyword(keyword="start recording"):
    """Listens for a specific keyword to start recording."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for the keyword...")
        # while keyboard.is_pressed('enter'):
        # # Adjust for ambient noise
        # recognizer.adjust_for_ambient_noise(source)
        # audio = recognizer.listen(source)
        # print("Recognizing...")

        # # Recognize speech using Google Web Speech API
        # command = recognizer.recognize_google(audio).lower()
        # print(f"Detected: {command}")

        # # Check if the command contains the keyword
        # if keyword.lower() in command:
        # print(f"Keyword '{keyword}' detected. Starting recording...")
        # break

        # except sr.UnknownValueError:
        #     print("Could not understand the audio, trying again.")
        # except sr.RequestError:
        #     print("Could not request results from Google Speech Recognition service.")
        #     break
