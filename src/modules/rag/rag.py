import json
import os
import traceback
import uuid
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator

import httpx
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue
from ray import serve
from ray.serve import handle

from src.core.module import ModuleWithHandle, ModuleWithId
from src.modules.text_to_speech.events import Token

from .events import RAGQuestion
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
    
    memory_collection: str = "conversations"
    memory_top_k: int = 3
    memory_half_life_days: float = 5.0
    memory_w_relevance: float = 0.5
    memory_w_recency: float = 0.3
    memory_w_importance: float = 0.2
    memory_maintenance_days: float = 5.0
    memory_maintenance_check_hours: float = 6.0


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

    _MAINTENANCE_MARKER_ID = "00000000-0000-0000-0000-00000000feed"
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
        if not hasattr(self, "_maintenance_task") or self._maintenance_task.done():
            self._maintenance_task = asyncio.get_event_loop().create_task(
                self._maintenance_loop()
            )

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
                        FieldCondition(
                            key="_user_id", match=MatchValue(value=_user_id)
                        ),
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

    def _ensure_memory_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams
        names = [c.name for c in self._qdrant.get_collections().collections]
        if self._cfg.memory_collection not in names:
            self._qdrant.create_collection(
                collection_name=self._cfg.memory_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    async def _llm_complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        """Non-streamed convenience wrapper over _llm_stream."""
        parts = []
        async for d in self._llm_stream(system_prompt, user_prompt, {"max_length": max_tokens}, None):
            parts.append(d)
        return "".join(parts)

    def _memory_strength(self, payload: dict, relevance: float) -> float:
        importance = payload.get("importance", 3)
        half_life = max(self._cfg.memory_half_life_days * (importance / 5.0), 0.5)
        try:
            last = datetime.fromisoformat(payload.get("last_accessed") or payload["created_at"])
            age_days = (datetime.now() - last).total_seconds() / 86400.0
        except Exception:
            age_days = 0.0
        recency = 0.5 ** (age_days / half_life)
        cfg = self._cfg
        return (cfg.memory_w_relevance * relevance
                + cfg.memory_w_recency * recency
                + cfg.memory_w_importance * (importance / 10.0))

    def _search_memories(self, query_vector: list[float], _user_id: str) -> list[str]:
        """Retrieve, re-rank (relevance+recency+importance), reinforce, return texts."""
        try:
            hits = self._qdrant.query_points(
                collection_name=self._cfg.memory_collection,
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key="_user_id", match=MatchValue(value=_user_id))],
                    must_not=[FieldCondition(key="type", match=MatchValue(value="maintenance_marker"))],
                ),
                limit=10,
                score_threshold=0.2,   # permissive; real filtering is the re-rank
            ).points
        except Exception:
            return []   # collection missing / qdrant down → just no memories

        scored = sorted(hits, key=lambda p: self._memory_strength(p.payload, p.score), reverse=True)
        top = scored[: self._cfg.memory_top_k]

        # MemoryBank-style reinforcement: recalled memories decay slower.
        now = datetime.now().isoformat()
        for p in top:
            try:
                self._qdrant.set_payload(
                    collection_name=self._cfg.memory_collection,
                    payload={"last_accessed": now,
                             "access_count": p.payload.get("access_count", 0) + 1},
                    points=[p.id],
                )
            except Exception:
                pass
        return [p.payload.get("text", "") for p in top if p.payload.get("text")]

    async def save_conversation(self, _user_id: str, history: list) -> None:
        """Summarize a finished session into one memory point. Called at disconnect."""
        if not history or len(history) < 2:
            return
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)[-6000:]
        prompt = (
            "Summarize this conversation in 2-4 sentences, keeping any personal "
            "facts, preferences, names, or commitments mentioned. Then rate 1-10 "
            "how important it is to remember (small talk=1-2, personal facts or "
            "preferences=7-10). Reply ONLY with JSON, no markdown: "
            '{"summary": "...", "importance": N}\n\n' + transcript
        )
        try:
            raw = await self._llm_complete("You are a memory summarizer.", prompt)
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(cleaned)
            summary = str(data["summary"])
            importance = max(1, min(int(data["importance"]), 10))
        except Exception:
            print(f"[RAG] memory summarization failed, storing raw tail:\n{traceback.format_exc()}")
            summary, importance = transcript[-500:], 3

        if importance <= 1:
            print("[RAG] Session judged not memorable, skipping save")
            return

        vector = await self._embed(summary)
        self._ensure_memory_collection(len(vector))
        now = datetime.now().isoformat()
        self._qdrant.upsert(
            collection_name=self._cfg.memory_collection,
            points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={
                "text": summary, "_user_id": _user_id, "type": "conversation",
                "created_at": now, "last_accessed": now,
                "access_count": 0, "importance": importance,
            })],
        )
        print(f"[RAG] Saved conversation memory (importance={importance}): {summary[:80]}...")


    def _last_maintenance(self) -> datetime | None:
        try:
            pts = self._qdrant.retrieve(
                collection_name=self._cfg.memory_collection,
                ids=[self._MAINTENANCE_MARKER_ID], with_payload=True, with_vectors=False,
            )
            if pts:
                return datetime.fromisoformat(pts[0].payload["last_run"])
        except Exception:
            pass
        return None

    def _mark_maintenance_done(self, vector_size: int) -> None:
        # Marker point: zero vector, type=maintenance_marker. Filtered out of
        # retrieval automatically (zero vector never scores) but be explicit anyway.
        self._qdrant.upsert(
            collection_name=self._cfg.memory_collection,
            points=[PointStruct(
                id=self._MAINTENANCE_MARKER_ID,
                vector=[0.0] * vector_size,
                payload={"type": "maintenance_marker",
                         "last_run": datetime.now().isoformat()},
            )],
        )

    async def _maintenance_loop(self) -> None:
        import asyncio
        check_secs = self._cfg.memory_maintenance_check_hours * 3600
        while True:
            try:
                last = self._last_maintenance()
                due = (last is None or
                       (datetime.now() - last).total_seconds()
                       >= self._cfg.memory_maintenance_days * 86400)
                if due:
                    print("[RAG] Running memory maintenance...")
                    await self._run_maintenance()
            except Exception:
                print(f"[RAG] maintenance loop error:\n{traceback.format_exc()}")
            await asyncio.sleep(check_secs)

    async def _run_maintenance(self) -> None:
        """Decay-based pruning + consolidation. Same logic as memory_maintenance.py."""
        from collections import defaultdict
        DELETE_BELOW, CONSOLIDATE_BELOW = 0.05, 0.30

        # scroll everything
        points, offset = [], None
        try:
            while True:
                batch, offset = self._qdrant.scroll(
                    collection_name=self._cfg.memory_collection, limit=200,
                    offset=offset, with_payload=True, with_vectors=False)
                points.extend(batch)
                if offset is None:
                    break
        except Exception:
            return   # collection doesn't exist yet — nothing to do

        def base_strength(payload: dict) -> float:
            # query-independent: recency * importance
            imp = payload.get("importance", 3)
            half = max(self._cfg.memory_half_life_days * (imp / 5.0), 0.5)
            try:
                last = datetime.fromisoformat(payload.get("last_accessed") or payload["created_at"])
                age = (datetime.now() - last).total_seconds() / 86400.0
            except Exception:
                age = 0.0
            return (0.5 ** (age / half)) * (imp / 10.0)

        to_delete, weak_by_user = [], defaultdict(list)
        vector_size = None
        for p in points:
            if p.payload.get("type") == "maintenance_marker":
                continue
            s = base_strength(p.payload)
            if s < DELETE_BELOW:
                to_delete.append(p)
            elif s < CONSOLIDATE_BELOW:
                weak_by_user[p.payload.get("_user_id", "anonymous")].append(p)

        for user, weak in weak_by_user.items():
            if len(weak) < 3:
                continue
            texts = [p.payload["text"] for p in weak]
            merged = (await self._llm_complete(
                "You are a memory consolidator.",
                "These are old memories about conversations with the same person. "
                "Merge them into a single 3-5 sentence memory keeping only durable "
                "facts, preferences and recurring themes. Drop one-off small talk.\n\n"
                + "\n---\n".join(texts))).strip()
            if not merged:
                continue
            vec = await self._embed(merged)
            vector_size = len(vec)
            now = datetime.now().isoformat()
            imp = min(max(p.payload.get("importance", 3) for p in weak) + 1, 10)
            self._qdrant.upsert(collection_name=self._cfg.memory_collection,
                points=[PointStruct(id=str(uuid.uuid4()), vector=vec, payload={
                    "text": merged, "_user_id": user,
                    "type": "conversation_consolidated",
                    "created_at": now, "last_accessed": now,
                    "access_count": 0, "importance": imp,
                })])
            to_delete.extend(weak)
            print(f"[RAG] Consolidated {len(weak)} memories → 1 for user {user}")

        if to_delete:
            self._qdrant.delete(collection_name=self._cfg.memory_collection,
                points_selector=PointIdsList(points=[p.id for p in to_delete]))
            print(f"[RAG] Deleted {len(to_delete)} decayed memories")

        if vector_size is None:
            vector_size = len(await self._embed("marker"))
        self._ensure_memory_collection(vector_size)
        self._mark_maintenance_done(vector_size)
        print("[RAG] Memory maintenance complete")

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
        memories: list[str] | None = None,
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
            "Use the context and memories in the user's message to inform your "
            "answers when relevant, but always answer in character. If you have "
            "relevant memories of past conversations, use them naturally. Only if "
            "you genuinely know nothing relevant, improvise in character rather "
            "than admitting you lack information or breaking character. "        )
        system_prompt = " ".join(parts)

        memory_block = ""
        if memories:
            memory_block = (
                "Things you remember from previous conversations with this person:\n- "
                + "\n- ".join(memories)
                + "\n\n"
            )

        if not chunks:
            user_prompt = (
                memory_block
                + "No relevant context was found.\n\n"
                f"Question: {question}\n\n"
                "Answer based on your memories above if relevant, otherwise general knowledge."
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
                memory_block
                + f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the context and your memories above. "
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
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (
                    chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
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
        memories = self._search_memories(query_vector, query._user_id)
        if memories:
            print(f"[RAG] Recalled {len(memories)} memory(ies)")
        profile_facts = self._get_profile(collection, query._user_id)
        if profile_facts:
            print(f"[RAG] Loaded {len(profile_facts)} profile fact(s)")
        system_prompt, user_prompt = self._build_prompt(
            query.question, chunks, query.preferences, profile_facts, memories
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

    input:  question (RAGQuestion)
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

        print(
            f"[RAG] Initialized with user_id={_user_id}, language={language}, tone={tone}, response_format={response_format}, max_length={max_length}, temperature={temperature}, max_history_turns={max_history_turns}"
        )

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

    async def process(self, data: RAGQuestion) -> AsyncGenerator[Token, None]:  # type: ignore[override]
        """
        Called when a "question" event arrives through the event bus.
        Packages _user_id + question, sends to the stateless RAGHandle.
        """
        question_text = data.transcript.text

        query = RAGQuery(
            _user_id=self._user_id if self._user_id else "anonymous",
            question=question_text,
            preferences=self.preferences,
            history=list(self.history),  # snapshot of prior turns
        )

        parts: list[str] = []
        stream = self._handle.options(stream=True).stream.remote(query)
        async for delta in stream:
            parts.append(delta)
            yield Token(text=delta, end=False)
        yield Token(text="", end=True)

        self._record_turn(question_text, "".join(parts))

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
    
    async def finalize(self) -> None:
        """Called when the session ends — persist this conversation as a memory."""
        if not self.history:
            return
        try:
            await self._handle.save_conversation.remote(
                self._user_id or "anonymous", list(self.history)
            )
        except Exception:
            print(f"[RAG] finalize failed:\n{traceback.format_exc()}")
