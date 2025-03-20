from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
import json
import pathlib

class Rag:
    def __init__(self, model="deepseek-r1:7b",
                 collectionName="vectorStore",
                 vectorstorePath="src/rag/vectorStore"):
        self.memory = MemorySaver()
        self.embeddings = OllamaEmbeddings(model=model)
        self.llm = OllamaLLM(model=model)
        self.vectorstore = Chroma(collection_name=collectionName,
                                  embedding_function=self.embeddings,
                                  persist_directory=vectorstorePath)
        self.textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.retriever = self.vectorstore.as_retriever()
        self.systemPrompt = "Conversation history:\n{history}\n\nContext:\n{context}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.systemPrompt),
                ("human", "{input}"),
            ]
        )
        self.questionChain = create_stuff_documents_chain(self.llm, self.prompt)
        self.qaChain = create_retrieval_chain(self.retriever, self.questionChain)
        self.documents = []
        self.docs = []
        self.conversation = []
        self.conversation_log = {"conversation": []}


    def ragLoader(self, text: str):
        self.documents += self.textSplitter.split_documents(text)
        self.vectorstore.add_documents(self.documents)       
    
    def ragLoader(self, pathFolder: str, fileType: str):
        if fileType == "txt":
            for file in pathlib.Path(pathFolder).rglob("*.txt"):
                fileLoader = TextLoader(file_path=pathFolder + "/" + file.name)
                self.documents += self.textSplitter.split_documents(fileLoader.load())
        self.vectorstore.add_documents(self.documents)       

    def ragQuestion(self, question: str):
        history = "\n".join([f"Human: {qa['question']}\nAI: {qa['answer']}" for qa in self.conversation_log["conversation"]])
        helpingContext = "Answer with just your message like in a conversation. "
        question = helpingContext + question
        response = self.qaChain.invoke({"history": history, "input": question})
        answer = response["answer"]
        self.conversation_log["conversation"].append({"question": question.split(helpingContext)[1:], "answer": answer})
        return answer
    
    def saveConversation(self, filename="conversation_log.json"):
        with open(filename, "w") as f:
            json.dump(self.conversation_log, f, indent=4)
