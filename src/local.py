import soundfile as sf
from llm.llm import get_chain
from speech_to_text.stt import get_whisper_model
from text_to_speech.tts import get_tts_model, tokenize_text, get_tts_tokenizer
import io
import time

# 🎙️ Process Audio (LOCAL)
def process_audio_local(audio_buffer: io.BytesIO) -> io.BytesIO:

    # Load models
    ollama = get_chain()
    whisper = get_whisper_model("base.en")
    tts = get_tts_model()
    tts_tokenizer = get_tts_tokenizer()

    # Speech-to-Text
    audio_buffer.seek(0)  # Ensure we're at the start
    audio_array, sample_rate = sf.read(audio_buffer, dtype="float32")
    # debug_audio_path = "debug_input.wav"
    # sf.write(debug_audio_path, audio_array, sample_rate)
    # print(f"✅ Debug: Saved input audio to {debug_audio_path}, {sample_rate}")
    print(f"📝 Transcription beginning...")
    start_time = time.perf_counter()
    prompt = whisper.transcribe(audio_array, language="en")["text"]
    elapsed_time = time.perf_counter() - start_time
    print(f"📝 Transcription: {prompt}, it took {elapsed_time:.2f}s")

    # Process text with LLM
    answer = ollama.invoke({"question": prompt})
    print(f"🤖 LLM Response: {answer}")

    # Convert LLM text to speech
    prompt_input_ids = tokenize_text(answer.split("</think>")[-1], tts_tokenizer)
    input_ids = tokenize_text("The voice of a young lady", tts_tokenizer)
    generation = tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    # Save output audio
    output_buffer = io.BytesIO()
    sf.write(output_buffer, audio_arr, tts.config.sampling_rate, format="WAV")
    output_buffer.seek(0)
    print(f"🎵 Audio generated")

    return output_buffer


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
