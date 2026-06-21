from typing import Any, Optional
from dataclasses import dataclass, field
 
from ray import data, serve
from ray.serve import handle
from src.core.module import ModuleWithHandle
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
 
 
import httpx
 
  
@dataclass
class RAGQuery:
    """What flows from RAG module to RAGHandle."""
    user_id: str
    question: str
    preferences: dict = field(default_factory=dict)
    # preferences can include: language, tone, response_format, max_length, system_prompt, extra_instructions, etc.
 
 
@dataclass
class RAGResult:
    """What RAGHandle returns."""
    answer: str
    sources: list[dict] = field(default_factory=list)
 
 
@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1},
)
class RAGHandle:
    """
    Stateless RAG processor. Knows nothing about sessions.
    Receives a user_id + question, uses user_id to find the right
    collection/data in the vector DB, runs embed -> search -> LLM.
    """
 
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        default_collection: str = "documents",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        llm_provider: str = "ollama", # "vllm", "ollama", "api"
        llm_url: str = "http://localhost:11434",
        llm_model: str = "mistral:7b",
        llm_api_key: str = "",
        top_k: int = 5,
        score_threshold: float = 0.5,
    ):
        self.embed_model = SentenceTransformer(embedding_model)
        self.qdrant = QdrantClient(url=qdrant_url)
        self.default_collection = default_collection
        self.top_k = top_k
        self.score_threshold = score_threshold
 
        self.llm_provider = llm_provider
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
 
    def _resolve_user_context(self, user_id: str) -> tuple[str, dict | None]:
        """
        Given a user_id, decide which collection to search
        and which filters to apply.
 
        Options (pick what fits your data model):
          A) One collection per user:  collection = f"user_{user_id}"
          B) Shared collection, filter by user_id in payload
          C) Lookup in a DB to find the user's config
        """
 
        # Option A: separate collection per user
        # collection = f"user_{user_id}"
        # filters = None
 
        # Option B: shared collection with user_id filter (recommended)
        collection = self.default_collection
        filters = {"user_id": user_id}
 
        return collection, filters


    def _embed(self, text) -> list[float]:
        return self.embed_model.encode(str(text), normalize_embeddings=True).tolist()



    def _search(
        self,
        query_vector: list[float],
        collection: str,
        filters: dict | None = None,
    ) -> list[dict]:
 
        # Build qdrant filter from user context
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        ).points

        return [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            }
            for point in results
        ]

    def _build_prompt(
        self,
        question: str,
        chunks: list[dict],
        preferences: dict,
    ) -> tuple[str, str]:

        parts = [
            "You are a helpful assistant. Answer based on the provided context.",
            "If the context is insufficient, say so clearly.",
        ]
        if preferences.get("language"):
            parts.append(f"Always respond in {preferences['language']}.")
        if preferences.get("tone"):
            parts.append(f"Use a {preferences['tone']} tone.")
        if preferences.get("response_format") == "bullet_points":
            parts.append("Format your answer as bullet points.")
        elif preferences.get("response_format") == "short":
            parts.append("Keep your answer to 2-3 sentences maximum.")
        if preferences.get("extra_instructions"):
            parts.append(preferences["extra_instructions"])
        system_prompt = " ".join(parts)
 
        if not chunks:
            user_prompt = (
                "No relevant context was found.\n\n"
                f"Question: {question}\n\n"
                "Answer based on general knowledge and mention no documents were found."
            )
        else:
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                source = chunk["metadata"].get("source", "unknown")
                context_parts.append(
                    f"[{i}] (source: {source}, score: {chunk['score']:.2f})\n{chunk['text']}"
                )
            context_block = "\n\n".join(context_parts)
            user_prompt = (
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the context above. Cite sources by number."
            )
 
        return system_prompt, user_prompt


    async def _llm_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        preferences: dict,
    ) -> str:
        max_tokens = preferences.get("max_length", 1024)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
 
        if self.llm_provider == "vllm":
            return await self._call_openai_compatible(
                f"{self.llm_url}/v1/chat/completions", messages, max_tokens
            )
        elif self.llm_provider == "ollama":
            return await self._call_ollama(messages, max_tokens)
        elif self.llm_provider == "api":
            return await self._call_openai_compatible(
                f"{self.llm_url}/v1/chat/completions", messages, max_tokens, self.llm_api_key
            )
        else:
            raise ValueError(f"Unknown llm_provider: {self.llm_provider}")


    async def _call_openai_compatible(
        self, url: str, messages: list, max_tokens: int, api_key: str = ""
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json={
                "model": self.llm_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            })
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


    async def _call_ollama(self, messages: list, max_tokens: int) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.llm_url}/api/chat", json={
                "model": self.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            })
            resp.raise_for_status()
            return resp.json()["message"]["content"]

 
    async def process(self, query: RAGQuery) -> RAGResult:
        """
        Main entry point. Called by the RAG module.
        Uses user_id to determine which collection / filters to use.
        """

        print(f"[RAG] Question: {query.question}")
        collection, filters = self._resolve_user_context(query.user_id) 
        query_vector = self._embed(query.question)
        chunks = self._search(query_vector, collection, filters)


        print(f"[RAG] Found {len(chunks)} chunks")
        for c in chunks:
            print(f"  - score: {c['score']:.2f} | {c['text'][:100]}...")

        system_prompt, user_prompt = self._build_prompt(
            query.question, chunks, query.preferences
        )
        print(f"[RAG] System prompt: {system_prompt[:200]}...") 
        answer = await self._llm_generate(system_prompt, user_prompt, query.preferences)
        print(f"[RAG] Answer: {answer}")

        return RAGResult(
            answer=answer,
            sources=[
                {"text": c["text"], "score": c["score"], "metadata": c["metadata"]}
                for c in chunks
            ],
        )
    
 
class RAG(ModuleWithHandle):
    """
    Session-bound module. HuRI instantiates this when a client connects,
    passing the user_id from the WebSocket config.
 
    Listens to "question" events.
    Forwards question + user_id to the detached RAGHandle.
    Emits "rag_response" event with the answer.
    """
    _handle_cls = RAGHandle
    input_type = "question"
    output_type = "rag_response"

    def __init__(
        self,
        handle: handle.DeploymentHandle[RAGHandle],
        user_id: str = "",
        language: str = "en",
        tone: str = "formal",
        response_format: str = "paragraph",
        max_length: int = 1024,
        extra_instructions: str = "",
    ):
        super().__init__(handle)
        self.user_id = user_id
        self.preferences = {
            "language": language,
            "tone": tone,
            "response_format": response_format,
            "max_length": max_length,
            "extra_instructions": extra_instructions,
        }
 
    async def process(self, data) -> Optional[Any]:
        """
        Called when a "question" event arrives through the event bus.
        Packages user_id + question, sends to the stateless RAGHandle.
        """
        question_text = data.text if hasattr(data, 'text') else str(data)

        query = RAGQuery(
            user_id=self.user_id if self.user_id else "anonymous",
            question=question_text,
            preferences=self.preferences,
        )
 
        result: RAGResult = await self.handle.process.remote(query)
        return result
 
    def update_preferences(self, new_preferences: dict):
        """Client can update preferences mid-session via the event bus."""
        self.preferences.update(new_preferences)
