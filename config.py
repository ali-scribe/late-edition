"""Configuration constants and environment settings for late-edition."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Gemini API Key (read from environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def get_default_model() -> str:
    """Get the model ID configured in .env or environment."""
    model = os.getenv("GEMMA_MODEL", "").strip()
    if not model or model == "gemma-4":
        raise ValueError("Set GEMMA_MODEL in .env, copy the exact id from Google AI Studio")
    return model


def __getattr__(name: str):
    """Dynamic attribute access for DEFAULT_MODEL ensuring no fake default."""
    if name == "DEFAULT_MODEL":
        return get_default_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Chronicling America / Library of Congress rate limiting (max 20 requests/minute)
LOC_RATE_LIMIT_PER_MINUTE = 20

# Cache directory (never refetch cached files)
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))

# Gemini API retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_FACTOR = 2.0

# Chunking settings
DEFAULT_CHUNK_SIZE = 800

# Guard threshold
MIN_SIMILARITY_RATIO = 0.80
