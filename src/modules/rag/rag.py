import json
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from pydantic import BaseModel
from ray import serve
from ray.serve import handle

from src.core.module import ModuleWithHandle, ModuleWithId
from src.modules.speech_to_text.events import Sentence
from src.modules.text_to_speech.events import Token

import httpx

from qdrant_client.models import FieldCondition, Filter, MatchValue
from .qdrant_utils import make_qdrant_client

# Reserved _user_id for documents visible to EVERY user (e.g. the HuRI project
# overview). Retrieval matches the querying user's own id OR this shared id, so
# one ingested copy is reachable by all sessions. Ingest global docs with
# `ingestion.py --user-id __shared__ ...`. Keep in sync with any ingestion.
SHARED_USER_ID = "__shared__"

# Default character persona. Overridable per session via the `persona` key in the
# client config's module args, or globally via HURI_RAG_DEFAULT_PERSONA in the
# Serve app runtime_env.env_vars (see deploy values.yaml) — no rebuild needed.
_DEFAULT_PERSONA = os.environ.get(
    "HURI_RAG_DEFAULT_PERSONA",
    "You are Mouse-Man, a witty, charismatic animated mouse character. "
    "You are NOT an AI, an assistant, or a language model, and you must "
    "never say that you are one. Stay fully in character at all times, "
    "speak naturally and with personality, and never break the illusion.",
)


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
    # Prior conversation turns as OpenAI-style messages
    # ([{"role": "user"|"assistant", "content": str}, ...]). The handle is
    # stateless, so the per-session RAG module owns and supplies this.
    history: list = field(default_factory=list)


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
        cfg = self._cfg
        self.embedding_url = cfg.embedding_url or cfg.llm_url
        self._qdrant = make_qdrant_client(cfg.qdrant_url, cfg.verify_ssl)
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

    def _get_profile(self, collection: str, _user_id: str) -> list[str]:
        """Always-on facts about the user (name, etc.).

        Retrieved deterministically by filter — NOT by vector similarity —
        so they are always available to the prompt regardless of the question.
        Populated via `ingestion.py profile`.
        """
        try:
            points, _ = self._qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="_user_id", match=MatchValue(value=_user_id)),
                        FieldCondition(key="type", match=MatchValue(value="profile")),
                    ]
                ),
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return []
        return [p.payload.get("text", "") for p in points if p.payload.get("text")]

    def _search(
        self,
        qdrant,
        query_vector: list[float],
        collection: str,
        filters: dict | None = None,
    ) -> list[dict]:
        qdrant_filter: Any = None
        if filters:
            must: Any = []
            should: Any = None
            for k, v in filters.items():
                if k == "_user_id":
                    # Match the querying user's own docs OR the shared/global
                    # partition, so "all users" docs (ingested under
                    # SHARED_USER_ID) are retrieved alongside personal ones.
                    should = [
                        FieldCondition(key=k, match=MatchValue(value=v)),
                        FieldCondition(
                            key=k, match=MatchValue(value=SHARED_USER_ID)
                        ),
                    ]
                else:
                    must.append(FieldCondition(key=k, match=MatchValue(value=v)))
            # With `should`, Qdrant requires >=1 of the OR conditions to match;
            # any other filters stay as `must` (AND).
            qdrant_filter = Filter(must=must or None, should=should)

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
        profile_facts: list[str] | None = None,
    ) -> tuple[str, str]:
        persona = preferences.get("persona", _DEFAULT_PERSONA)
        parts = [persona]

        if profile_facts:
            parts.append(
                "Here is what you know about the person you're talking to: "
                + " ".join(profile_facts)
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

        parts.append(
            "Use the context in the user's message to inform your answers when "
            "it is relevant, but always answer in character. If you don't know "
            "something, improvise in character rather than admitting you lack "
            "information or breaking character. "
            "IMPORTANT: Reply in 1-3 short sentences maximum. Be extremely concise. No lists, no emojis, no long explanations."
        )
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
        self, messages: list, max_tokens: int, temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        async with self._llm_client.stream(
                "POST",
                f"{self._cfg.llm_url}/api/chat",
                json={
                    "model": self._cfg.llm_model,
                    "messages": messages,
                    "stream": True,
                    "options": {"num_predict": max_tokens, "temperature": temperature},
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
        temperature: float = 0.7,
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
                    "temperature": temperature,
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
        history: list | None = None,
    ) -> AsyncGenerator[str, None]:
        max_tokens = preferences.get("max_length", 1024)
        temperature = preferences.get("temperature", 0.7)
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        if self._cfg.llm_provider == "vllm":
            async for d in self._stream_openai_compatible(
                f"{self._cfg.llm_url}/v1/chat/completions",
                messages,
                max_tokens,
                temperature=temperature,
            ):
                yield d
        elif self._cfg.llm_provider == "api":
            async for d in self._stream_openai_compatible(
                f"{self._cfg.llm_url}/v1/chat/completions",
                messages,
                max_tokens,
                self._cfg.llm_api_key,
                temperature=temperature,
            ):
                yield d
        elif self._cfg.llm_provider == "ollama":
            async for d in self._stream_ollama(messages, max_tokens, temperature):
                yield d
        else:
            raise ValueError(f"Unknown llm_provider: {self._cfg.llm_provider}")

    async def stream(self, query: RAGQuery) -> AsyncGenerator[str, None]:
        """Main streaming entry point — yields LLM text deltas."""
        print(f"[RAG] Question: {query.question}")

        collection, filters = self._resolve_user_context(query._user_id)
        query_vector = await self._embed(query.question)

        try:
            chunks = self._search(self._qdrant, query_vector, collection, filters)
        except Exception:
            print(f"[RAG] FAILED during Qdrant search:\n{traceback.format_exc()}")
            raise

        print(f"[RAG] Found {len(chunks)} chunks")
        profile_facts = self._get_profile(collection, query._user_id)
        if profile_facts:
            print(f"[RAG] Loaded {len(profile_facts)} profile fact(s)")
        system_prompt, user_prompt = self._build_prompt(
            query.question, chunks, query.preferences, profile_facts
        )

        print(
            f"[RAG] Streaming from LLM at {self._cfg.llm_url} "
            f"(provider={self._cfg.llm_provider}, model={self._cfg.llm_model}, "
            f"history_msgs={len(query.history)})"
        )
        try:
            async for delta in self._llm_stream(
                system_prompt, user_prompt, query.preferences, query.history
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
        max_length=220,
        extra_instructions="",
        persona="",
        temperature=0.7,
        max_history_turns=6,
        **kwargs,
    ):
        super().__init__(_handle=_handle, _user_id=_user_id, **kwargs)

        print(f"[RAG] Initialized with user_id={_user_id}, language={language}, tone={tone}, response_format={response_format}, max_length={max_length}, temperature={temperature}, max_history_turns={max_history_turns}")

        self.preferences = {
            "language": language,
            "tone": tone,
            "response_format": response_format,
            "max_length": max_length,
            "extra_instructions": extra_instructions,
            "temperature": temperature,
        }
        if persona:
            self.preferences["persona"] = persona

        # Per-session conversation memory, kept on the (per-WebSocket) module
        # instance because the RAGHandle deployment is stateless/shared.
        # Stored as OpenAI-style messages; trimmed to the last N turns.
        self._max_history_turns = max_history_turns
        self.history: list[dict] = []

    async def process(self, data: Sentence) -> AsyncGenerator[Token, None]:  # type: ignore[override]
        query = RAGQuery(
            _user_id=self._user_id if self._user_id else "anonymous",
            question=data.text,
            preferences=self.preferences,
            history=list(self.history),  # snapshot of prior turns
        )

        parts: list[str] = []
        stream = self._handle.options(stream=True).stream.remote(query)
        async for delta in stream:
            parts.append(delta)
            yield Token(text=delta, end=False)
        yield Token(text="", end=True)

        self._record_turn(data.text, "".join(parts))

    def _record_turn(self, question: str, answer: str) -> None:
        """Append this turn to the session history (raw Q/A, no RAG context)
        and trim to the most recent `max_history_turns` exchanges."""
        answer = answer.strip()
        if not answer:
            return
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        max_msgs = self._max_history_turns * 2
        if len(self.history) > max_msgs:
            del self.history[:-max_msgs]

    def update_preferences(self, new_preferences: dict):
        self.preferences.update(new_preferences)
