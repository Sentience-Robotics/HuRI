import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from pydantic import BaseModel
from ray import serve
from ray.serve import handle

from src.core.module import ModuleWithHandle, ModuleWithId
from src.modules.speech_to_text.events import Sentence
from src.modules.text_to_speech.events import Token


class RAGDeploymentConfig(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    default_collection: str = "documents"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_url: str = ""
    llm_provider: str = "ollama"  # "vllm", "ollama", "api"
    llm_url: str = "http://localhost:11434"
    llm_model: str = "mistral:7b"
    llm_api_key: str = ""
    verify_ssl: bool = True
    top_k: int = 5
    score_threshold: float = 0.5


@dataclass
class RAGQuery:
    """What flows from RAG module to RAGHandle."""

    _user_id: str
    question: str
    preferences: dict = field(default_factory=dict)


@serve.deployment(name="RAGHandle")
class RAGHandle:
    """Stateless RAG processor. Streams LLM tokens to the caller."""

    def __init__(self, **kwargs):
        self._cfg = RAGDeploymentConfig(**kwargs)
        self._apply_config()

    def reconfigure(self, config: dict) -> None:
        self._cfg = RAGDeploymentConfig(**{**self._cfg.model_dump(), **config})
        self._apply_config()

    def _apply_config(self) -> None:
        import httpx
        from qdrant_client import QdrantClient

        cfg = self._cfg
        self.embedding_url = cfg.embedding_url or cfg.llm_url
        self._qdrant = QdrantClient(url=cfg.qdrant_url, verify=cfg.verify_ssl)
        print(f"[RAGHandle] Connected to Qdrant at {cfg.qdrant_url}")
        self._embed_client = httpx.AsyncClient(timeout=30.0, verify=cfg.verify_ssl)
        self._llm_client = httpx.AsyncClient(timeout=120.0, verify=cfg.verify_ssl)

    def _resolve_user_context(self, _user_id: str) -> tuple[str, dict | None]:
        collection = self._cfg.default_collection
        filters = {"_user_id": _user_id}
        return collection, filters

    async def _embed(self, text: str) -> list[float]:
        url = f"{self.embedding_url}/v1/embeddings"
        resp = await self._embed_client.post(
            url,
            json={"model": self._cfg.embedding_model, "input": str(text)},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding HTTP {resp.status_code} from {url}: {resp.text[:1000]}"
            )
        try:
            payload = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"Embedding non-JSON response from {url}: {resp.text[:1000]}"
            ) from e
        try:
            return payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Embedding unexpected schema from {url}: {str(payload)[:1000]}"
            ) from e

    def _search(
        self,
        qdrant,
        query_vector: list[float],
        collection: str,
        filters: dict | None = None,
    ) -> list[dict]:
        qdrant_filter: Any = None
        if filters:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            conditions: Any = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        try:
            results = qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=self._cfg.top_k,
                score_threshold=self._cfg.score_threshold,
            ).points
        except Exception:
            results = []
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
            "You are a robot speaking to a user. Answer based on the provided context.",
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
                "Answer based on general knowledge."
            )
        else:
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                source = chunk["metadata"].get("source", "unknown")
                context_parts.append(
                    f"[{i}] (source: {source}, score: {chunk['score']:.2f})\n"
                    f"{chunk['text']}"
                )
            context_block = "\n\n".join(context_parts)
            user_prompt = (
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the context above. "
                "Don't speak about the sources, just use them to answer."
            )

        return system_prompt, user_prompt

    async def _stream_ollama(
        self, messages: list, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        async with self._llm_client.stream(
                "POST",
                f"{self._cfg.llm_url}/api/chat",
                json={
                    "model": self._cfg.llm_model,
                    "messages": messages,
                    "stream": True,
                    "options": {"num_predict": max_tokens, "temperature": 0.1},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        return

    async def _stream_openai_compatible(
        self,
        url: str,
        messages: list,
        max_tokens: int,
        api_key: str = "",
    ) -> AsyncGenerator[str, None]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with self._llm_client.stream(
                "POST",
                url,
                headers=headers,
                json={
                    "model": self._cfg.llm_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield delta

    async def _llm_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        preferences: dict,
    ) -> AsyncGenerator[str, None]:
        max_tokens = preferences.get("max_length", 1024)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self._cfg.llm_provider == "vllm":
            async for d in self._stream_openai_compatible(
                f"{self._cfg.llm_url}/v1/chat/completions", messages, max_tokens
            ):
                yield d
        elif self._cfg.llm_provider == "api":
            async for d in self._stream_openai_compatible(
                f"{self._cfg.llm_url}/v1/chat/completions",
                messages,
                max_tokens,
                self._cfg.llm_api_key,
            ):
                yield d
        elif self._cfg.llm_provider == "ollama":
            async for d in self._stream_ollama(messages, max_tokens):
                yield d
        else:
            raise ValueError(f"Unknown llm_provider: {self._cfg.llm_provider}")

    async def stream(self, query: RAGQuery) -> AsyncGenerator[str, None]:
        """Main streaming entry point — yields LLM text deltas."""
        import traceback

        print(f"[RAG] Question: {query.question}")

        collection, filters = self._resolve_user_context(query._user_id)
        query_vector = await self._embed(query.question)

        try:
            chunks = self._search(self._qdrant, query_vector, collection, filters)
        except Exception:
            print(f"[RAG] FAILED during Qdrant search:\n{traceback.format_exc()}")
            raise

        print(f"[RAG] Found {len(chunks)} chunks")
        system_prompt, user_prompt = self._build_prompt(
            query.question, chunks, query.preferences
        )

        print(f"[RAG] Streaming from LLM at {self._cfg.llm_url} (provider={self._cfg.llm_provider}, model={self._cfg.llm_model})")
        try:
            async for delta in self._llm_stream(
                system_prompt, user_prompt, query.preferences
            ):
                yield delta
        except Exception:
            print(f"[RAG] FAILED during LLM stream:\n{traceback.format_exc()}")
            raise


class RAG(ModuleWithHandle, ModuleWithId):
    """RAG Module — streams LLM tokens.

    input:  question (Sentence)
    output: token    (Token)
    """

    _handle_cls = RAGHandle
    input_type = "question"
    output_type = "token"

    def __init__(
        self,
        _handle: handle.DeploymentHandle,
        _user_id: str,
        language="en",
        tone="formal",
        response_format="paragraph",
        max_length=1024,
        extra_instructions="",
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

    async def process(self, data: Sentence) -> AsyncGenerator[Token, None]:  # type: ignore[override]
        query = RAGQuery(
            _user_id=self._user_id if self._user_id else "anonymous",
            question=data.text,
            preferences=self.preferences,
        )

        stream = self._handle.options(stream=True).stream.remote(query)
        async for delta in stream:
            yield Token(text=delta, end=False)
        yield Token(text="", end=True)

    def update_preferences(self, new_preferences: dict):
        self.preferences.update(new_preferences)
