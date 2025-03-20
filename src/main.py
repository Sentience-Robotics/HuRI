import os

from enum import Enum
import soundfile as sf
import simpleaudio as sa
from speech_to_text.speech_to_text import SpeechToText
from text_to_speech.tts import TextToSpeech
from rag.rag import Rag


class Modes(Enum):
    EXIT = 0
    LLM = 1
    CONTEXT = 2
    RAG = 3


def loop(stt: SpeechToText, tts: TextToSpeech, mode: Modes, mode_function: dict):
    while mode:
        prompt = stt.get_prompt()
        print(prompt)
        if "switch llm" in prompt.lower():
            mode = Modes.LLM
        elif "switch context" in prompt.lower():
            mode = Modes.CONTEXT
        elif "switch rag" in prompt.lower():
            mode = Modes.RAG
        elif "bye bye" in prompt.lower():
            mode = Modes.EXIT
        elif prompt.strip() == "":
            continue
        else:
            stt.pause()
            answer = mode_function[mode](prompt)
            tts.speak(answer, language='en', speaker_wav=['ref_basile.wav'])
            stt.pause(False)


def main():
    stt = SpeechToText()
    tts = TextToSpeech("tts_models--multilingual--multi-dataset--xtts_v2")
    rag = Rag(model="deepseek-v2:16b")
    rag.ragLoader("tests/rag/docsRag", "txt")
    mode = Modes.LLM
    mode_function = {
        Modes.LLM: rag.ragQuestion,
        Modes.RAG: rag.ragLoader,
        Modes.CONTEXT: lambda x: "Context mode not implemented yet.",
    }
    stt.start()
    try:
        loop(stt, tts, mode, mode_function)
    except KeyboardInterrupt:
        print("CTRL+C detected. Stopping the program.")
    except Exception as e:
        print("Unexpected Error:", e)
    stt.stop()


if __name__ == "__main__":
    main()
