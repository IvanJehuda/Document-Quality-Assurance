"""LLM provider selection (Google Gemini / Groq / Ollama) driven by environment config."""

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
# Max output tokens per page call — Groq requires explicit limit; Gemini defaults to model max.
GROQ_VISION_MAX_TOKENS = int(os.getenv("GROQ_VISION_MAX_TOKENS", "8192"))
GEMINI_VISION_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_VISION_MAX_OUTPUT_TOKENS", "0"))  # 0 = model default

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Controls which provider handles vision/image tasks (PDF pages).
# "google" → Gemini (default), "groq" → llama-4-scout via Groq API.
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "google").lower()

_PROVIDER_KEYS = {"google": GOOGLE_API_KEY, "groq": GROQ_API_KEY, "ollama": None}

if LLM_PROVIDER not in _PROVIDER_KEYS:
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'google', 'groq', or 'ollama'.")

if VISION_PROVIDER not in ("google", "groq"):
    raise RuntimeError(f"Unsupported VISION_PROVIDER '{VISION_PROVIDER}'. Use 'google' or 'groq'.")

# Ollama runs locally and needs no API key - only cloud providers are checked here.
if LLM_PROVIDER != "ollama":
    _active_key = _PROVIDER_KEYS[LLM_PROVIDER]
    if not _active_key or _active_key.startswith("your_"):
        raise RuntimeError(
            f"{LLM_PROVIDER.upper()}_API_KEY is missing or still a placeholder. "
            "Set it in your .env file before starting the server."
        )


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if LLM_PROVIDER == "groq":
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, groq_api_key=GROQ_API_KEY)
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature, base_url=OLLAMA_BASE_URL)
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )


def get_vision_llm() -> BaseChatModel:
    """Return a vision-capable LLM for PDF image extraction.

    Controlled by VISION_PROVIDER in .env:
      "google" (default) → Gemini via GOOGLE_API_KEY
      "groq"             → llama-4-scout via GROQ_API_KEY

    Raises RuntimeError if the required API key for the selected provider is missing.
    """
    if VISION_PROVIDER == "groq":
        if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
            raise RuntimeError(
                "VISION_PROVIDER=groq requires GROQ_API_KEY in .env."
            )
        return ChatGroq(
            model=GROQ_VISION_MODEL,
            temperature=0.0,
            groq_api_key=GROQ_API_KEY,
            max_tokens=GROQ_VISION_MAX_TOKENS,
            max_retries=0,  # our pdf_extraction.py handles retries with semaphore + backoff
        )

    # Default: google / Gemini
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("your_"):
        raise RuntimeError(
            "VISION_PROVIDER=google requires GOOGLE_API_KEY in .env. "
            "Set VISION_PROVIDER=groq to use Groq llama-4-scout instead."
        )
    gemini_kwargs = {}
    if GEMINI_VISION_MAX_OUTPUT_TOKENS > 0:
        gemini_kwargs["max_output_tokens"] = GEMINI_VISION_MAX_OUTPUT_TOKENS
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
        **gemini_kwargs,
    )
