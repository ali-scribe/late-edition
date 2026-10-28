"""Chunking module: splits OCR text into chunks of ~800 characters on paragraph/sentence boundaries."""

import re
from typing import Optional
from config import DEFAULT_CHUNK_SIZE


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences along terminal punctuation (.!?) followed by whitespace."""
    # Handle common abbreviations like Mr., Mrs., Dr., etc. from falsely breaking sentences
    # Split on sentence boundaries: positive lookbehind for .!? followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()] if text.strip() else []


def _split_large_segment(segment: str, max_chars: int) -> list[str]:
    """Split a segment that has no clean sentence boundaries into sub-chunks on clause/space boundaries."""
    if len(segment) <= max_chars:
        return [segment]
    
    parts = []
    # Try splitting on clause punctuation like ; , or newline
    clauses = re.split(r"(?<=[;,\n])\s+", segment)
    current = ""
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) > max_chars:
            # Fall back to word boundaries
            words = clause.split()
            word_accum = ""
            for word in words:
                if word_accum and len(word_accum) + 1 + len(word) > max_chars:
                    parts.append(word_accum.strip())
                    word_accum = word
                else:
                    word_accum = f"{word_accum} {word}" if word_accum else word
            if word_accum:
                parts.append(word_accum.strip())
        elif current and len(current) + 1 + len(clause) > max_chars:
            parts.append(current.strip())
            current = clause
        else:
            current = f"{current} {clause}" if current else clause
    if current:
        parts.append(current.strip())
    return parts


def split_into_chunks(text: str, target_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Split OCR text into chunks of about target_size characters on paragraph or sentence boundaries.
    
    Prefers splitting on paragraph boundaries (\n\n), then sentence boundaries ([.!?]),
    and falls back to clauses or word boundaries when necessary.
    """
    if not text or not text.strip():
        return []

    # Clean carriage returns
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Step 1: Split into paragraphs
    raw_paragraphs = re.split(r"\n\s*\n+", normalized_text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        # If paragraph fits into current chunk comfortably
        p_len = len(paragraph)

        # Break paragraph into sentences
        sentences = _split_into_sentences(paragraph)

        for sentence in sentences:
            s_len = len(sentence)
            # If a single sentence is exceedingly large (> target_size * 1.2), break it down
            if s_len > int(target_size * 1.2):
                sub_parts = _split_large_segment(sentence, target_size)
            else:
                sub_parts = [sentence]

            for part in sub_parts:
                part_len = len(part)
                # Check if adding this part would significantly exceed target size
                # We allow some slack (e.g. if current_chunk is small, keep it; if >= 400 chars, start new chunk)
                if current_chunk and (current_len + part_len + 1 > target_size) and current_len >= 350:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [part]
                    current_len = part_len
                else:
                    current_chunk.append(part)
                    current_len += part_len + (1 if current_chunk else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
