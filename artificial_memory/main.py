import sys

from prompt import prompt
from sanitize import sanitize_response

if __name__ == "__main__":
    with open('pre_prompt.txt', 'r') as file:
        pre_prompt = file.read()
        pre_prompt = pre_prompt.replace('\n', ' ')
    if len(sys.argv) > 1:
        coded_prompt = sys.argv[1]
    else:
        coded_prompt = "User: Hello my name is Matthias, I am 78 years old. You: My name is Sigma Boy, I am a robot."

    response = prompt(pre_prompt, coded_prompt)
    sanitized_response = sanitize_response(response)
    print(sanitized_response)