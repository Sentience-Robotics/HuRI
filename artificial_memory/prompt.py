from ollama import chat
from ollama import ChatResponse

def prompt(pre_prompt: str, prompt: str) -> str:
    prompt = f"{pre_prompt} {prompt}"

    response: ChatResponse = chat(model='mistral', messages=[{
        'role': 'user',
        'content': prompt,
    }])
    return response['message']['content']
