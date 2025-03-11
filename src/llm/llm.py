from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

def get_llm() -> OllamaLLM:
    model = OllamaLLM(model="deepseek-r1:7b")
    return model
    
def get_template() -> ChatPromptTemplate:
    temp = """Question: {question}

    Answer: quick answer like a txt file':."""

    template = ChatPromptTemplate.from_template(temp)
    return template

def get_chain():
    chain = get_template() | get_llm()
    return chain
