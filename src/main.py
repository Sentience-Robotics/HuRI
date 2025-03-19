import os

from llm.llm import get_chain
from speech_to_text.stt import get_whisper_model
from speech_to_text.recorder import listen_for_keyword
from text_to_speech.tts import get_tts_model, tokenize_text, get_tts_tokenizer
import soundfile as sf
import simpleaudio as sa

def main():
    ollama = get_chain()
    whisper = get_whisper_model("base.en")
    tts = get_tts_model()
    tts_tokenizer = get_tts_tokenizer()
    while True:
        listen_for_keyword("start recording")
        prompt = whisper.transcribe("test_audio/my_audio.wav", language='en')["text"]
        print(prompt)
        answer = ollama.invoke({"question": prompt})
        print(answer)
        if prompt == "exit":
            break
        prompt_input_ids = tokenize_text(answer.split("</think>")[-1], tts_tokenizer)
        input_ids = tokenize_text("The voice of Donald Trump", tts_tokenizer)
        generation = tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write("test_audio/parler_tts_out.wav", audio_arr, tts.config.sampling_rate)
        print("Audio saved as test_audio/parler_tts_out.wav")

        print("Playing audio...")
        wave_obj = sa.WaveObject.from_wave_file("test_audio/parler_tts_out.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        print("Audio played")
        
        
if __name__ == "__main__":
    main()