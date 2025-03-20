from TTS.api import TTS
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play

class TextToSpeech:
    def __init__(self, model_name: str) -> None:
        self.tts = TTS(
            model_name=model_name.replace('--', '/')
        )
        print("audio device:", sd.default.device)

    def __play_sound(self, path: str) -> None:
        song = AudioSegment.from_wav(path)
        play(song)

    def speak(self, text: str) -> None:
        path = self.tts.tts_to_file(
            text=text,
            file_path='output.wav'
        )
        self.__play_sound(path)

    def __del__(self) -> None:
        sd.wait()
