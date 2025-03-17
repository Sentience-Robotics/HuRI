import os

IS_MODAL = os.getenv("MODAL_ENVIRONMENT") == "1"

if not IS_MODAL:
    import sounddevice as sd
    import soundfile as sf
    import wave
    from local import process_audio_local
    import io

from modal_fct import modal_main

# File paths
INPUT_WAV = "test_audio/input.wav"
OUTPUT_WAV = "test_audio/output.wav"


# 🎤 Record Audio
def record_audio(duration=5, sample_rate=16000): #sample rate must be 16000 so that whisper can transcribe it
    print("🎙️ Recording... Speak now!")
    recorded_audio = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16"
    )
    sd.wait()

    # with wave.open(file_path, "wb") as wf:
    #     wf.setnchannels(1)
    #     wf.setsampwidth(2)
    #     wf.setframerate(sample_rate)
    #     wf.writeframes(recording.tobytes())
    input_buffer = io.BytesIO()
    sf.write(input_buffer, recorded_audio, sample_rate, format="WAV")
    print(f"✅ Recording done.")
    return input_buffer


def play_audio(output_buffer: io.BytesIO):
    output_buffer.seek(0)
    data, samplerate = sf.read(output_buffer)
    print(f"🎵 Playing audio")
    sd.play(data, samplerate)
    sd.wait()  # Wait until audio finishes playing
    print("✅ Audio playback complete.")


# 🎧 Main Execution
def main():
    audio_buffer = record_audio(duration=5)

    choice = input("🔄 Process audio locally (L) or with Modal (M)? ").strip().lower()

    if choice == "m":
        output_buffer = modal_main(audio_buffer)
    else:
        output_buffer = process_audio_local(audio_buffer)

    play_audio(output_buffer)


if __name__ == "__main__":
    main()


# import os

# from llm.llm import get_chain
# from speech_to_text.stt import get_whisper_model
# from speech_to_text.recorder import listen_for_keyword
# from text_to_speech.tts import get_tts_model, tokenize_text, get_tts_tokenizer
# from speech_to_text.audio import speech_to_wav

# def main():
#     ollama = get_chain()
#     whisper = get_whisper_model("base.en")
#     tts = get_tts_model()
#     tts_tokenizer = get_tts_tokenizer()
#     while True:
#         speech_to_wav("test_audio/my_speech.wav")
#         prompt = whisper.transcribe("test_audio/my_speech.wav", language='en')["text"]
#         print(prompt)
#         answer = ollama.invoke({"question": prompt})
#         print(answer)
#         if prompt == "exit":
#             break
#         prompt_input_ids = tokenize_text(answer.split("</think>")[-1], tts_tokenizer)
#         input_ids = tokenize_text("The voice of Donald Trump", tts_tokenizer)
#         generation = tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
#         audio_arr = generation.cpu().numpy().squeeze()
#         sf.write("test_audio/parler_tts_out.wav", audio_arr, tts.config.sampling_rate)
#         print("Audio saved as test_audio/parler_tts_out.wav")

#         print("Playing audio...")
#         wave_obj = sa.WaveObject.from_wave_file("test_audio/parler_tts_out.wav")
#         play_obj = wave_obj.play()
#         play_obj.wait_done()
#         print("Audio played")

#         # os.remove("test_audio/my_audio.wav")
#         # os.remove("test_audio/parler_tts_out.wav")


# if __name__ == "__main__":
#     main()
