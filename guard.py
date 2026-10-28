"""Guard module: validates Gemma 4 corrections against original OCR chunks using difflib.

Rejection criteria (reverts to original chunk):
(a) Any digit changed
(b) A new capitalized word appears that is not in the original chunk
(c) Similarity ratio is below 0.80

Every edit is recorded as an EditHunk with edit type and accepted/flagged status.
Flagged hallucinations / new capitalized words are surfaced as 'hauntings'.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from config import MIN_SIMILARITY_RATIO


@dataclass
class EditHunk:
    """Represents a discrete text diff edit between original and corrected text."""
    type: str  # 'replace', 'insert', 'delete'
    original: str
    corrected: str
    status: str  # 'accepted' or 'flagged'
    reason: Optional[str] = None
    orig_start: int = 0
    orig_end: int = 0
    corr_start: int = 0
    corr_end: int = 0


@dataclass
class GuardResult:
    """Result of running guard validation on an OCR chunk."""
    accepted: bool
    final_text: str
    original_text: str
    corrected_text: str
    similarity_ratio: float
    flag_reasons: list[str] = field(default_factory=list)
    hunks: list[EditHunk] = field(default_factory=list)
    hauntings: list[str] = field(default_factory=list)


def extract_digits(text: str) -> list[str]:
    """Extract all individual digit characters from text."""
    return re.findall(r"\d", text)


def extract_capitalized_words(text: str) -> list[str]:
    """Extract capitalized words (tokens beginning with an uppercase letter)."""
    return re.findall(r"\b[A-Z][a-zA-Z0-9_\'-]*\b", text)


def extract_all_word_tokens(text: str) -> list[str]:
    """Extract all word tokens from text."""
    return re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text)


# Common closed-class grammatical words that commonly begin sentences or clauses
# and should not be falsely flagged as hallucinated named-entity hauntings when fixing OCR typos (e.g. 'Thc' -> 'The')
COMMON_GRAMMATICAL_WORDS = {
    "The", "A", "An", "And", "But", "Or", "Nor", "For", "Yet", "So",
    "In", "On", "At", "To", "By", "With", "From", "Of", "About", "Into",
    "Through", "After", "Before", "Under", "Above", "Between", "It", "Its",
    "He", "His", "Him", "She", "Her", "Hers", "They", "Them", "Their",
    "Theirs", "We", "Us", "Our", "Ours", "You", "Your", "Yours",
    "This", "That", "These", "Those", "There", "Here", "Where", "When",
    "Why", "How", "What", "Which", "Who", "Whom", "Whose", "If", "Then",
    "Than", "Though", "Although", "Because", "Since", "Unless", "While",
    "As", "Each", "Every", "Both", "All", "Any", "Some", "No", "Not",
    "Only", "Very", "More", "Most", "Other", "Another", "Such",
}


def find_new_capitalized_words(original: str, corrected: str) -> list[str]:
    """Identify any capitalized words in corrected that were not present in the original chunk.
    
    A word is considered already present if:
    1. It is a standard grammatical word / sentence starter (e.g. 'The', 'In', 'And' fixing 'Thc')
    2. It appeared in original chunk in exact case or capitalized form (e.g. 'Washington' -> 'Washington')
    3. It appeared in uppercase in original chunk (e.g. 'WASHINGTON' -> 'Washington' in headlines)
    4. It appeared in original chunk in lowercase (e.g. capitalizing sentence starts: 'the' -> 'The')
    
    Any newly invented capitalized entity (e.g. hallucinating 'Lincoln' or 'London') is flagged.
    """
    corr_cap_words = extract_capitalized_words(corrected)
    orig_tokens = extract_all_word_tokens(original)
    
    # Fast lookup sets
    orig_token_set = set(orig_tokens)
    orig_cap_set = {w for w in orig_tokens if w[0].isupper()}
    orig_lower_set = {w.lower() for w in orig_tokens}

    new_capitals = []
    for word in corr_cap_words:
        # Check if the word is a common grammatical word
        if word in COMMON_GRAMMATICAL_WORDS:
            continue
        # Check if the word was already present in original in some form
        if word in orig_cap_set:
            continue
        if word.upper() in orig_cap_set:
            continue
        if word.lower() in orig_lower_set:
            continue
        # If neither exact, all-caps, nor lower was present, it's a new capitalized word
        new_capitals.append(word)

    # Return unique preserved in order of appearance
    seen = set()
    unique_new = []
    for w in new_capitals:
        if w not in seen:
            seen.add(w)
            unique_new.append(w)
    return unique_new


def validate_chunk(
    original: str,
    corrected: str,
    min_similarity: float = MIN_SIMILARITY_RATIO,
) -> GuardResult:
    """Compare original OCR chunk against corrected chunk using difflib.
    
    Reject and revert if:
    (a) Any digit changed
    (b) A new capitalized word appears that is not in the original chunk
    (c) Similarity ratio is below min_similarity (default 0.80)
    
    Records every edit as an EditHunk with type and accepted/flagged status.
    """
    # 1. Compute similarity ratio using difflib
    matcher = difflib.SequenceMatcher(None, original, corrected)
    similarity = matcher.ratio()

    flag_reasons: list[str] = []
    hauntings: list[str] = []

    # Check (a): Digit changes
    orig_digits = extract_digits(original)
    corr_digits = extract_digits(corrected)
    if orig_digits != corr_digits:
        flag_reasons.append(
            f"digit_changed (original digits: {''.join(orig_digits)!r}, "
            f"corrected digits: {''.join(corr_digits)!r})"
        )

    # Check (b): New capitalized words (hauntings)
    new_capitals = find_new_capitalized_words(original, corrected)
    if new_capitals:
        hauntings.extend(new_capitals)
        flag_reasons.append(f"new_capitalized_word: {', '.join(new_capitals)}")

    # Check (c): Similarity ratio
    if similarity < min_similarity:
        flag_reasons.append(
            f"similarity_below_threshold ({similarity:.3f} < {min_similarity:.2f})"
        )

    # Acceptance determination
    accepted = len(flag_reasons) == 0
    final_text = corrected if accepted else original

    # Record every edit hunk
    hunks: list[EditHunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        orig_slice = original[i1:i2]
        corr_slice = corrected[j1:j2]

        hunk_status = "accepted" if accepted else "flagged"
        hunk_reasons = []

        if not accepted:
            # Attribute specific reasons to this hunk if applicable
            if extract_digits(orig_slice) != extract_digits(corr_slice):
                hunk_reasons.append("digit_changed")
            hunk_new_caps = [w for w in new_capitals if w in corr_slice]
            if hunk_new_caps:
                hunk_reasons.append(f"haunting: {', '.join(hunk_new_caps)}")
            if similarity < min_similarity:
                hunk_reasons.append("low_similarity")
            if not hunk_reasons:
                hunk_reasons.append("chunk_rejected")

        hunk_reason_str = "; ".join(hunk_reasons) if hunk_reasons else None

        hunks.append(
            EditHunk(
                type=tag,
                original=orig_slice,
                corrected=corr_slice,
                status=hunk_status,
                reason=hunk_reason_str,
                orig_start=i1,
                orig_end=i2,
                corr_start=j1,
                corr_end=j2,
            )
        )

    return GuardResult(
        accepted=accepted,
        final_text=final_text,
        original_text=original,
        corrected_text=corrected,
        similarity_ratio=similarity,
        flag_reasons=flag_reasons,
        hunks=hunks,
        hauntings=hauntings,
    )
