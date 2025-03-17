import modal
import io
import socket
import time

from local import process_audio_local

# Define Modal stub
stub = modal.Stub("speech-processing")

# Define Modal image with required dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg", "curl")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install_from_requirements("requirements_modal.txt")
)

MODAL_GPU = "A10G"

# 🎙️ Process Audio (MODAL)
@stub.function(image=image, gpu=MODAL_GPU, timeout=600, keep_warm=1)
def process_audio_modal(audio_bytes: bytes) -> bytes:
    import subprocess
    print("🚀 Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", 11434))
        if result == 0:
            print("✅ Ollama is running and accepting connections!")
            break
        else:
            print("⏳ Waiting for Ollama to start...")
            time.sleep(2)

    print("🚀 Pulling model 'deepseek-r1:7b'...")
    subprocess.run(["ollama", "pull", "deepseek-r1:7b"], check=True)
    print("✅ 'deepseek-r1:7b' is pulled!")
    input_buffer = io.BytesIO(audio_bytes)
    output_buffer = process_audio_local(input_buffer)
    return output_buffer.getvalue()

def modal_main(audio_buffer: io.BytesIO) -> io.BytesIO:
    with stub.run():
        output_bytes = process_audio_modal.remote(audio_buffer.getvalue())
        return io.BytesIO(output_bytes)

