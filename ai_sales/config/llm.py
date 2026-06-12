"""
LLM configuration and initialization for the AI Sales Agent.

Handles:
- Environment variable loading (AGENTS.md: Always run load_dotenv() at the top)
- Lazy LLM instantiation to avoid import-time crashes
- API key resolution: supports both GOOGLE_API_KEY and GEMINI_API_KEY
- Tool binding (AGENTS.md: Run llm.bind_tools(tools) before invoking)
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# AGENTS.md: Always run load_dotenv() at the top
load_dotenv()

# ---------------------------------------------------------------------------
# Lazy LLM Singleton
# ---------------------------------------------------------------------------
_llm_cache: dict = {}


def resolve_api_key() -> str:
    """Resolve the Google API key from environment variables.

    Supports both GOOGLE_API_KEY and GEMINI_API_KEY (which is what
    Google AI Studio provides by default).

    Raises:
        ValueError: If no API key is found in environment variables.
    """
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Google API key found. "
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file. "
            "Get a key at: https://aistudio.google.com/apikey"
        )
    return key


def get_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Return a base LLM instance (without tools), cached per temperature.

    Args:
        temperature: Sampling temperature. Lower = more deterministic (better
            for reliable tool calling); higher = more creative (replies/scoring).
    """
    key = f"base:{temperature}"
    if key not in _llm_cache:
        _llm_cache[key] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            google_api_key=resolve_api_key(),
        )
    return _llm_cache[key]


# Low temperature for the sales agent: factual, tool-driven replies only.
# Gemini at higher temperature tends to guess ("มีหลายแบบให้เลือก") before tools run.
_TOOL_LLM_TEMPERATURE = 0.1


def get_llm_with_tools(tools: list):
    """Return the (low-temperature) LLM with tools bound.

    AGENTS.md: Run llm.bind_tools(tools) before invoking.

    Args:
        tools: List of @tool-decorated functions to bind.
    """
    if "with_tools" not in _llm_cache:
        _llm_cache["with_tools"] = get_llm(_TOOL_LLM_TEMPERATURE).bind_tools(tools)
    return _llm_cache["with_tools"]


def get_llm_with_tools_forced(tools: list):
    """Like ``get_llm_with_tools`` but forces the model to call a tool.

    Used as a safety net: if the agent merely *announces* that it will search
    (without emitting a tool call), we re-invoke with ``tool_choice="any"`` so a
    tool actually runs and the graph can complete the turn with real results.
    """
    if "with_tools_forced" not in _llm_cache:
        _llm_cache["with_tools_forced"] = get_llm(_TOOL_LLM_TEMPERATURE).bind_tools(
            tools, tool_choice="any"
        )
    return _llm_cache["with_tools_forced"]
