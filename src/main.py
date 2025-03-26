from enum import Enum
import soundfile as sf
import simpleaudio as sa
from emotional_hub.input_analysis import predict_emotion
from speech_to_text.speech_to_text import SpeechToText
from rag.rag import Rag


class Modes(Enum):
    EXIT = 0
    LLM = 1
    CONTEXT = 2
    RAG = 3


def loop(stt: SpeechToText, tts: None, mode: Modes, mode_function):
    while mode:
        prompt, audio = stt.get_prompt()
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
            emotion = predict_emotion(audio)
            print("Predicted Emotion:", emotion)
            answer = mode_function[mode](f"in a {emotion} emotion: {prompt}")
            print(answer)
            stt.pause(False)


def main():
    stt = SpeechToText()
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
        loop(stt, None, mode, mode_function)
    except KeyboardInterrupt:
        print("CTRL+C detected. Stopping the program.")
    except Exception as e:
        print("Unexpected Error:", e)
    stt.stop()


if __name__ == "__main__":
    main()
