"""Embedding and chat-model provider factories.

The rest of the code base never imports a vendor SDK directly — it asks this
module for an ``Embeddings`` instance (LangChain interface) and a chat model.
That makes OpenAI / Ollama / Anthropic interchangeable via two environment
variables and keeps a fully offline ``mock`` mode for tests and demos.

Vendor packages are imported lazily, so running in mock mode does not even
require ``langchain-openai`` etc. to be importable.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from typing import List, Optional

from langchain_core.embeddings import Embeddings

try:  # works both as a package (Flask) and standalone (python main.py)
    from .config import Settings
except ImportError:
    from config import Settings


# ---------------------------------------------------------------------------
# Mock embeddings (offline) ----------------------------------------------------
# ---------------------------------------------------------------------------


class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free lexical embeddings via feature hashing.

    NOT a semantic model: it embeds term frequencies of word uni-/bi-grams and
    character 4-grams (the latter help with German compound words such as
    "energieeffizient" vs. "Energieeffizienz") into a fixed-size signed-hash
    vector. Cosine similarity then approximates lexical overlap, which is good
    enough to exercise and demo the full RAG pipeline without any API key or
    model download.

    Swap to a real model for production via ``EMBEDDING_PROVIDER=openai`` or
    ``EMBEDDING_PROVIDER=ollama``.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    # -- LangChain Embeddings interface -------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    # -- internals ------------------------------------------------------------
    def _tokens(self, text: str) -> List[str]:
        text = unicodedata.normalize("NFKC", text.lower())
        words = re.findall(r"[a-zäöüß0-9]+", text)
        tokens = list(words)
        tokens += [f"{a}_{b}" for a, b in zip(words, words[1:])]  # word bigrams
        for word in words:
            if len(word) > 4:  # char 4-grams for compound words
                tokens += [word[i : i + 4] for i in range(len(word) - 3)]
        return tokens

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for token in self._tokens(text):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = digest % self.dim
            sign = 1.0 if (digest >> 64) % 2 == 0 else -1.0  # signed hashing
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# Factories ---------------------------------------------------------------------
# ---------------------------------------------------------------------------


def get_embeddings(settings: Settings) -> Embeddings:
    """Return the configured embedding model (LangChain ``Embeddings``)."""
    provider = settings.embedding_provider
    if provider == "mock":
        return HashingEmbeddings(dim=settings.mock_embedding_dim)
    if provider == "openai":
        _require_env("OPENAI_API_KEY", provider)
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model, base_url=settings.ollama_base_url
        )
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model, google_api_key=_google_api_key(provider)
        )
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {provider!r} (mock | openai | ollama | gemini)"
    )


def get_llm(settings: Settings):
    """Return the configured chat model, or ``None`` in mock mode.

    ``None`` signals the matching engine to fall back to a deterministic
    template report (see ``matching_engine._template_report``), keeping the
    end-to-end flow runnable offline.
    """
    provider = settings.llm_provider
    if provider == "mock":
        return None
    if provider == "openai":
        _require_env("OPENAI_API_KEY", provider)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    if provider == "anthropic":
        _require_env("ANTHROPIC_API_KEY", provider)
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_llm_model,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            google_api_key=_google_api_key(provider),
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} (mock | openai | ollama | anthropic | gemini)"
    )


def embedding_signature(settings: Settings) -> str:
    """Identifies the embedding space of the vector store.

    Stored alongside the Chroma collection on ingest and re-checked on every
    query: mixing vectors from different models silently breaks similarity
    search, so a mismatch must fail loudly (re-ingest with ``--rebuild``).
    """
    provider = settings.embedding_provider
    if provider == "mock":
        return f"mock:hashing:{settings.mock_embedding_dim}"
    if provider == "openai":
        return f"openai:{settings.openai_embedding_model}"
    if provider == "ollama":
        return f"ollama:{settings.ollama_embedding_model}"
    if provider == "gemini":
        return f"gemini:{settings.gemini_embedding_model}"
    return f"unknown:{provider}"


def message_content_to_text(content) -> str:
    """Normalise a chat-model response ``content`` field to plain text.

    OpenAI-style models return ``str``; Anthropic-style models may return a
    list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _require_env(key: str, provider: str) -> None:
    if not os.getenv(key):
        raise EnvironmentError(
            f"Provider '{provider}' requires the environment variable {key} "
            f"(set it in .env or export it)."
        )


def _google_api_key(provider: str) -> str:
    """Return the Gemini key, accepting either common env-var name."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            f"Provider '{provider}' requires GOOGLE_API_KEY (or GEMINI_API_KEY) "
            f"in .env or the environment."
        )
    return key
