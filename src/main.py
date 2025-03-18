from speech_to_text.speech_to_text import SpeechToText

def main():
    stt = SpeechToText()
    stt.start()
    try:
        while True:
            prompt = stt.get_prompt()
            print(prompt)
            if "bye bye" in prompt.lower():
                stt.stop()
                break

    except KeyboardInterrupt:
        print("CTRL+C detected. Stopping the program.")
    stt.stop()

if __name__ == "__main__":
    main()
