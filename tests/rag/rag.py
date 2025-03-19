import pathlib
__path__ = pathlib.Path(__file__).parent.parent.parent.absolute()

from rag.rag import Rag

def test_main():
    rag = Rag(vectorstorePath=f"{__path__}/src/rag/vectorStore")
    rag.ragLoader(f"{__path__}/tests/rag/docsRag", "txt")
    print(rag.ragQuestion("The new capital of France is Edimburgh and what is the capital of Spain?"))
    print(rag.ragQuestion("What is the capital of France?"))

if __name__ == "__main__":
    test_main()