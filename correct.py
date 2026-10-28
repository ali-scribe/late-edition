"""OCR text correction using Gemma 4 via the Google Gemini API with exponential backoff."""

import logging
import random
import time
from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

import config
from config import (
    BACKOFF_FACTOR,
    GEMINI_API_KEY,
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# Exact prompt per project specification:
PROMPT_INSTRUCTION = (
    "Fix OCR errors, spelling, spacing and broken words only; "
    "do not add, remove or reword content; "
    "do not change numbers; "
    "return only the corrected text."
)


def build_correction_prompt(chunk_text: str) -> str:
    """Construct prompt enclosing the chunk text with strict instructions."""
    return (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"Input text:\n{chunk_text}\n\n"
        f"Corrected text:"
    )


def correct_chunk_with_gemini(
    chunk_text: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Send an OCR text chunk to Gemma 4 via the Gemini API with exponential backoff on 429/5xx.
    
    Hard cap on retries ensures termination.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your .env file or environment."
        )

    model = model_name or config.DEFAULT_MODEL
    client = genai.Client(api_key=key)

    prompt = build_correction_prompt(chunk_text)
    attempt = 0

    while True:
        try:
            logger.debug(f"Calling Gemini API (model: {model}, attempt {attempt + 1}/{max_retries + 1})...")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=0.95,
                    max_output_tokens=2048,
                ),
            )
            corrected = response.text or ""
            # Clean possible markdown wrapping if model accidentally outputs code fence
            cleaned = corrected.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                lines = cleaned.splitlines()
                if len(lines) >= 2:
                    cleaned = "\n".join(lines[1:-1]).strip()
            return cleaned

        except Exception as exc:
            attempt += 1
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

            # Check if error is transient / retryable (429 rate limit or 5xx server errors)
            is_rate_limit = status_code == 429 or "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            is_server_error = (
                (isinstance(status_code, int) and status_code >= 500)
                or "500" in str(exc)
                or "503" in str(exc)
                or "UNAVAILABLE" in str(exc)
            )

            if (is_rate_limit or is_server_error) and attempt <= max_retries:
                # Exponential backoff with small random jitter
                backoff = (INITIAL_BACKOFF_SECONDS * (BACKOFF_FACTOR ** (attempt - 1))) + random.uniform(0.1, 0.5)
                logger.warning(
                    f"Gemini API returned status {status_code} on attempt {attempt}/{max_retries}. "
                    f"Retrying in {backoff:.2f}s... Error: {exc}"
                )
                time.sleep(backoff)
            else:
                logger.error(f"Gemini API call failed after {attempt} attempts: {exc}")
                raise exc


def mock_correct_chunk(chunk_text: str) -> str:
    """Mock correction function for offline testing, demos, or dry-runs without API keys.
    
    Fixes common OCR glitches while obeying all guard constraints.
    """
    import re
    # Fix common 19th century OCR glitches without inventing words or changing digits
    corrected = chunk_text
    corrections = [
        (r"\bthc\b", "the"),
        (r"\bThc\b", "The"),
        (r"\bwns\b", "was"),
        (r"\bvv\b", "w"),
        (r"\btbe\b", "the"),
        (r"\bteh\b", "the"),
        (r"\breceivcd\b", "received"),
        (r"\bReceivcd\b", "Received"),
        (r"\byestcrday\b", "yesterday"),
        (r"\bYestcrday\b", "Yesterday"),
        (r"\bgovernmcnt\b", "government"),
        (r"\bGovernmcnt\b", "Government"),
        (r"(\w+)-\s*\n\s*(\w+)", r"\1\2"),  # recombine broken hyphenated words across lines
    ]
    for pattern, repl in corrections:
        corrected = re.sub(pattern, repl, corrected)
    return corrected
