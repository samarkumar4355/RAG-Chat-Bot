"""
Central configuration for Free AI models and API Keys.
Provider keys are read from Streamlit secrets or environment variables.
"""

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _get_api_key(name: str) -> str:
    if value := os.getenv(name):
        return value

    try:
        return st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        return ""


API_KEYS = {
    "groq": _get_api_key("GROQ_API_KEY"),
    "openrouter": _get_api_key("OPENROUTER_API_KEY"),
    "gemini": _get_api_key("GEMINI_API_KEY"),
}

AVAILABLE_FREE_MODELS = {
    "⚡ Groq — GPT-OSS 120B (Recommended)": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "badge": "Groq · 120B",
        "description": "Ultra-fast & powerful 120B model on Groq hardware",
    },
    "⚡ Groq — Qwen 3.8 27B (High Accuracy)": {
        "provider": "groq",
        "model": "qwen/qwen3.8-27b",
        "badge": "Groq · Qwen",
        "description": "Strong reasoning & high precision document analysis",
    },
    "⚡ Groq — GPT-OSS 20B (Instant Speed)": {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "badge": "Groq · 20B",
        "description": "Lightweight model optimized for rapid response",
    },
    "🌐 OpenRouter — Nemotron 3.5 Lightning": {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3.5-lightning:free",
        "badge": "OpenRouter · Nvidia",
        "description": "Nvidia high-efficiency instruction-tuned model",
    },
    "🌐 OpenRouter — Ling 3.2 Flash": {
        "provider": "openrouter",
        "model": "inclusionai/ling-3.2-flash:free",
        "badge": "OpenRouter · Ling",
        "description": "Fast multilingual flash model",
    },
    "🌐 OpenRouter — Ernie 5.5 Flash": {
        "provider": "openrouter",
        "model": "baidu/ernie-5.5-flash-preview:free",
        "badge": "OpenRouter · Ernie",
        "description": "Baidu Ernie 5.5 preview",
    },
    "💎 Google — Gemini 3.8 Flash": {
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "badge": "Gemini · Flash",
        "description": "Google Gemini 3.8 Flash model",
    },
}

DEFAULT_MODEL = "⚡ Groq — GPT-OSS 120B (Recommended)"
