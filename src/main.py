import os

from enum import Enum
from llm.llm import get_chain
# from speech_to_text.recorder import listen_for_keyword
# from text_to_speech.tts import get_tts_model, tokenize_text, get_tts_tokenizer
import soundfile as sf
import simpleaudio as sa
from speech_to_text.speech_to_text import SpeechToText
from rag.rag import Rag

class Modes(Enum):
    EXIT = 0
    LLM = 1
    CONTEXT = 2
    RAG = 3

def loop(stt: SpeechToText, mode: Modes, mode_function: dict):
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
            answer = mode_function[mode](prompt)
            print(answer)
        
def main():
    stt = SpeechToText()
    rag = Rag(model="deepseek-v2:16b")
    rag.ragLoader("tests/rag/docsRag", "txt")
    mode = Modes.LLM
    mode_function = {
        Modes.LLM: rag.ragQuestion,
        Modes.RAG: rag.ragLoader
    }
    stt.start()
    try:
        loop(stt, mode, mode_function)
    except KeyboardInterrupt:
        print("CTRL+C detected. Stopping the program.")
    except Exception as e:
        print("Unexpected Error:", e)
    stt.stop()


if __name__ == "__main__":
    main()
