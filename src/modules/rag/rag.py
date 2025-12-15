import json
import pathlib

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver

from src.core.module import Module


class Rag(Module):
    def __init__(
        self,
        model: str = "deepseek-v2:16b",
        collectionName: str = "vectorStore",
        vectorstorePath: str = "src/rag/vectorStore",
    ):
        super().__init__()
        self.memory = MemorySaver()
        self.embeddings = OllamaEmbeddings(model=model)
        self.llm = OllamaLLM(model=model)
        self.vectorstore = Chroma(
            collection_name=collectionName,
            embedding_function=self.embeddings,
            persist_directory=vectorstorePath,
        )
        self.textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
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

    def ragFill(self, text: str) -> None:
        self.documents += self.textSplitter.split_documents(text)
        self.vectorstore.add_documents(self.documents)

    def ragLoad(self, folderPath: str, fileType: str) -> None:
        if fileType == "txt":
            for file in pathlib.Path(folderPath).rglob("*.txt"):
                fileLoader = TextLoader(file_path=folderPath + "/" + file.name)
                self.documents += self.textSplitter.split_documents(fileLoader.load())
        self.vectorstore.add_documents(self.documents)

    def ragQuestion(self, question: str) -> None:
        self.logger.debug("question:", question)
        history = "\n".join(
            [
                f"Human: {qa['question']}\nAI: {qa['answer']}"
                for qa in self.conversation_log["conversation"]
            ]
        )
        helpingContext = "Answer with just your message like in a conversation. "
        question = helpingContext + question
        self.logger.debug("full question:", question)
        response = self.qaChain.invoke({"history": history, "input": question})
        answer = response["answer"]
        self.logger.debug("answer:", answer)
        self.conversation_log["conversation"].append(
            {"question": question.split(helpingContext)[1:], "answer": answer}
        )
        self.publish("llm.response", answer)

    def saveConversation(self, filename: str = "conversation_log.json"):
        with open(filename, "w") as f:
            json.dump(self.conversation_log, f, indent=4)

    def set_subscriptions(self) -> None:
        self.subscribe("rag.load", self.ragLoad)
        self.subscribe("llm.in", self.ragQuestion)
        self.subscribe("rag.in", self.ragFill)
        self.subscribe("rag.save", self.saveConversation)

    def run_module(self, stop_event=None) -> None:
        self.ragLoad("tests/rag/docsRag", "txt")
        super().run_module(stop_event)
