from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from ray import serve
from ray.serve import handle
from sentence_transformers import SentenceTransformer

from src.core.module import ModuleWithHandle, ModuleWithId
from src.modules.speech_to_text.events import Sentence

from .events import RAGResult


@dataclass
class RAGQuery:
    """What flows from RAG module to RAGHandle."""

    _user_id: str
    question: str
    preferences: dict = field(default_factory=dict)
    history: list[dict] | None = None
    # preferences can include: language, tone,
    # response_format, max_length, system_prompt, extra_instructions, etc.


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1},
)
class RAGHandle:
    """
    Stateless RAG processor. Knows nothing about sessions.
    Receives a _user_id + question, uses _user_id to find the right
    collection/data in the vector DB, runs embed -> search -> LLM.
    """

    def __init__(
        self,
        ollama_handle=None,
        qdrant_handle=None,
        qdrant_url: str = "http://localhost:6333",
        default_collection: str = "documents",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        llm_provider: str = "ollama",  # "vllm", "ollama", "api"
        llm_url: str = "http://localhost:11434",
        llm_model: str = "mistral:7b",
        llm_api_key: str = "",
        top_k: int = 5,
        score_threshold: float = 0.5,
    ):
        self.embed_model = SentenceTransformer(embedding_model)
        self.default_collection = default_collection
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.llm_provider = llm_provider
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key

        self.ollama_handle = ollama_handle
        self.qdrant_handle = qdrant_handle

        self._qdrant_url = qdrant_url
        self._qdrant: QdrantClient | None = None

    async def _get_qdrant(self):
        """Connect to Qdrant on first use. Solves the async-in-init problem."""
        if self._qdrant is None:
            if self.qdrant_handle:
                self._qdrant_url = await self.qdrant_handle.get_url.remote()
            self._qdrant = QdrantClient(url=self._qdrant_url)
            print(f"[RAGHandle] Connected to Qdrant at {self._qdrant_url}")
        return self._qdrant

    def _resolve_user_context(self, _user_id: str) -> tuple[str, dict | None]:
        """
        Given a _user_id, decide which collection to search
        and which filters to apply.

        Options (pick what fits your data model):
          A) One collection per user:  collection = f"user_{_user_id}"
          B) Shared collection, filter by _user_id in payload
          C) Lookup in a DB to find the user's config
        """

        collection = self.default_collection
        filters = {"_user_id": _user_id}

        return collection, filters

    def _embed(self, text) -> list[float] | Any:
        return self.embed_model.encode(str(text), normalize_embeddings=True).tolist()

    def _search(
        self,
        qdrant,
        query_vector: list[float],
        collection: str,
        filters: dict | None = None,
    ) -> list[dict]:

        qdrant_filter: Any = None
        if filters:
            conditions: Any = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        doc_results = []
        try:
            doc_results = qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=self.top_k,
                score_threshold=self.score_threshold,
            ).points
        except Exception:
            pass

        return [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            }
            for point in doc_results
        ]

    def _build_prompt(
        self,
        question: str,
        chunks: list[dict],
        preferences: dict,
        history=None,
    ) -> tuple[str, str]:

        parts = []

        if history:
            lines = [f"{m['role']}: {m['content']}" for m in history]
            parts.append("[Recent conversation]\n" + "\n".join(lines))

        parts.append(
            "You are a robot speaking to a user. Answer based on the provided context."
            + " If the context is insufficient, say so clearly.",
        )
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
            user_prompt = f"Question: {question}\n\n"
        else:
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                source = chunk["metadata"].get("source", "unknown")
                context_parts.append(f"[{i}] (source: {source}, score: \
{chunk['score']:.2f})\n{chunk['text']}")
            context_block = "\n\n".join(context_parts)
            user_prompt = (
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the context above.\
Don't speak about the sources, just use them to answer the question."
            )

        return system_prompt, user_prompt

    async def _llm_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        preferences: dict,
    ) -> Any:
        max_tokens = preferences.get("max_length", 1024)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.ollama_handle:
            return await self.ollama_handle.generate.remote(messages, max_tokens)

        if self.llm_provider == "vllm":
            return await self._call_openai_compatible(
                f"{self.llm_url}/v1/chat/completions", messages, max_tokens
            )
        elif self.llm_provider == "ollama":
            return await self._call_ollama(messages, max_tokens)

        elif self.llm_provider == "api":
            return await self._call_openai_compatible(
                f"{self.llm_url}/v1/chat/completions",
                messages,
                max_tokens,
                self.llm_api_key,
            )
        else:
            raise ValueError(f"Unknown llm_provider: {self.llm_provider}")

    async def _call_openai_compatible(
        self, url: str, messages: list, max_tokens: int, api_key: str = ""
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                headers=headers,
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_ollama(self, messages: list, max_tokens: int) -> Any:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.llm_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.1},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def process(self, query: RAGQuery) -> RAGResult:
        """
        Main entry point. Called by the RAG module.
        Uses _user_id to determine which collection / filters to use.
        """

        print(f"[RAG] Question: {query.question}")

        qdrant = await self._get_qdrant()

        collection, filters = self._resolve_user_context(query._user_id)
        query_vector = self._embed(query.question)
        chunks = self._search(qdrant, query_vector, collection, filters)

        print(f"[RAG] Found {len(chunks)} chunks")
        for c in chunks:
            print(f"  - score: {c['score']:.2f} | {c['text'][:100]}...")

        system_prompt, user_prompt = self._build_prompt(
            query.question, chunks, query.preferences, query.history
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


class RAG(ModuleWithHandle, ModuleWithId):
    _handle_cls = RAGHandle
    input_type = "question"
    output_type = "rag_response"

    def __init__(
        self,
        _handle: handle.DeploymentHandle[RAGHandle],
        _user_id: str,
        language="en",
        tone="formal",
        response_format="paragraph",
        max_length=1024,
        extra_instructions="",
        max_history=10,
        **kwargs,
    ):
        super().__init__(_handle=_handle, _user_id=_user_id, **kwargs)

        self.preferences = {
            "language": language,
            "tone": tone,
            "response_format": response_format,
            "max_length": max_length,
            "extra_instructions": extra_instructions,
        }
        self.history: list[dict] = []
        self.max_history = max_history

    async def process(self, data: Sentence) -> Optional[RAGResult]:
        """
        Called when a "question" event arrives through the event bus.
        Packages _user_id + question, sends to the stateless RAGHandle.
        """
        question_text = data.text

        query = RAGQuery(
            _user_id=self._user_id if self._user_id else "anonymous",
            question=question_text,
            preferences=self.preferences,
            history=(
                self.history
                if len(self.history) <= self.max_history
                else self.history[-self.max_history :]
            ),
        )

        if self._handle is None:
            print("[RAG] No handle available, returning None")
            return None

        result: RAGResult | None = None
        if self._handle is not None:
            result = await self._handle.process.remote(query)

        self.history.append({"role": "user", "content": question_text})
        self.history.append(
            {"role": "assistant", "content": result.answer if result else None}
        )

        return result

    def update_preferences(self, new_preferences: dict):
        """Client can update preferences mid-session via the event bus."""
        self.preferences.update(new_preferences)
