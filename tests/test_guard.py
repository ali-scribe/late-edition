"""Comprehensive pytest test suite for guard.py."""

import pytest
import guard
from guard import (
    EditHunk,
    GuardResult,
    extract_capitalized_words,
    extract_digits,
    find_new_capitalized_words,
    validate_chunk,
)


class TestDigitChecks:
    def test_extract_digits(self):
        assert extract_digits("Issue 42, Jan 1901, Page 3") == ["4", "2", "1", "9", "0", "1", "3"]
        assert extract_digits("No digits here!") == []

    def test_digit_modified_rejected(self):
        original = "The expedition returned in 1898 with four specimens."
        corrected = "The expedition returned in 1899 with four specimens."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert any("digit_changed" in r for r in result.flag_reasons)
        assert all(h.status == "flagged" for h in result.hunks)

    def test_digit_added_rejected(self):
        original = "Section A outlines the agreement reached."
        corrected = "Section 2 outlines the agreement reached."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert any("digit_changed" in r for r in result.flag_reasons)

    def test_digit_removed_rejected(self):
        original = "The 3 sailors survived the gale."
        corrected = "The sailors survived the gale."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert any("digit_changed" in r for r in result.flag_reasons)

    def test_digit_reordered_rejected(self):
        original = "Room 12 was locked."
        corrected = "Room 21 was locked."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert any("digit_changed" in r for r in result.flag_reasons)

    def test_digits_preserved_accepted(self):
        original = "On May 15, 1902, thc mayor spokc to 500 citizens."
        corrected = "On May 15, 1902, the mayor spoke to 500 citizens."
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert result.final_text == corrected
        assert len(result.flag_reasons) == 0


class TestCapitalizedWordsChecks:
    def test_extract_capitalized_words(self):
        text = "The Mayor of Chicago visited London yesterday."
        assert extract_capitalized_words(text) == ["The", "Mayor", "Chicago", "London"]

    def test_hallucinated_name_rejected_and_recorded_as_haunting(self):
        original = "The general gave orders to advance at dawn."
        corrected = "The general Grant gave orders to advance at dawn."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert "Grant" in result.hauntings
        assert any("new_capitalized_word" in r for r in result.flag_reasons)

    def test_hallucinated_place_rejected(self):
        original = "A special meeting was held by the council."
        corrected = "A special meeting was held in Boston by the council."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert "Boston" in result.hauntings

    def test_existing_capitalized_words_accepted(self):
        original = "Governor Roosevelt arrived in Albany on Wednesday morning."
        corrected = "Governor Roosevelt arrived in Albany on Wednesday morning."
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert len(result.hauntings) == 0

    def test_all_caps_headline_preserved_accepted(self):
        original = "THE EVENING WORLD REPORTS GREAT STORM on thc coast."
        corrected = "THE EVENING WORLD REPORTS GREAT STORM on the coast."
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert len(result.hauntings) == 0

    def test_sentence_initial_word_capitalization_accepted(self):
        original = "yesterday afternoon the committee adjourned."
        corrected = "Yesterday afternoon the committee adjourned."
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert len(result.hauntings) == 0


class TestSimilarityRatioChecks:
    def test_similarity_ratio_below_threshold_rejected(self):
        original = "The weather was bright and sunny across all counties yesterday."
        # Completely different text yielding similarity well below 0.80
        corrected = "A terrible explosion took place at the factory killing nobody."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert result.similarity_ratio < 0.80
        assert any("similarity_below_threshold" in r for r in result.flag_reasons)

    def test_similarity_ratio_high_accepted(self):
        original = "Thc quick brown fox jumpcd ovcr the lazy dog."
        corrected = "The quick brown fox jumped over the lazy dog."
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert result.similarity_ratio >= 0.80
        assert result.final_text == corrected


class TestEditHunksAndStatus:
    def test_exact_match_no_edits(self):
        text = "This is a clean and identical string with no OCR errors."
        result = validate_chunk(text, text)
        assert result.accepted
        assert result.similarity_ratio == 1.0
        assert len(result.hunks) == 0
        assert len(result.flag_reasons) == 0

    def test_accepted_edits_hunk_types(self):
        # replace 'c' -> 'e' in 'thc' -> 'the'
        original = "thc cat sat"
        corrected = "the cat sat"
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert len(result.hunks) == 1
        hunk = result.hunks[0]
        assert hunk.type == "replace"
        assert hunk.original == "c"
        assert hunk.corrected == "e"
        assert hunk.status == "accepted"
        assert hunk.reason is None

    def test_flagged_edits_hunks_record_reasons(self):
        original = "The treaty was signed in 1918."
        corrected = "The treaty was signed in 1919 by Woodrow Wilson."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        assert len(result.hunks) > 0
        assert all(h.status == "flagged" for h in result.hunks)
        assert "Woodrow" in result.hauntings
        assert "Wilson" in result.hauntings


class TestMultipleViolations:
    def test_multiple_violations_records_all_reasons(self):
        original = "The year was 1850."
        corrected = "The year was 1900 in New York according to modern records."
        result = validate_chunk(original, corrected)
        assert not result.accepted
        assert result.final_text == original
        # Both digit change and new capitalized words should be flagged
        reasons_text = " ".join(result.flag_reasons)
        assert "digit_changed" in reasons_text
        assert "new_capitalized_word" in reasons_text


class TestEdgeCases:
    def test_empty_strings(self):
        result = validate_chunk("", "")
        assert result.accepted
        assert result.final_text == ""
        assert len(result.hunks) == 0

    def test_only_whitespace_changes(self):
        original = "The  weather   was  cold."
        corrected = "The weather was cold."
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert result.final_text == corrected

    def test_punctuation_and_spelling_restore(self):
        original = "thc ship arrivcd at port, safc and sound"
        corrected = "the ship arrived at port, safe and sound."
        result = validate_chunk(original, corrected)
        assert result.accepted
        assert result.final_text == corrected
        assert any(h.type == "replace" for h in result.hunks)


class TestPlausibleRepairs:
    @pytest.mark.parametrize(
        ("candidate", "chunk"),
        [
            ("Houdini", "The magician Houdlni performed."),
            ("President", "The Presi dent addressed the crowd."),
            ("Monday", "The meeting was held Mon- day."),
        ],
    )
    def test_plausible_repairs(self, candidate, chunk):
        predicate = getattr(guard, "is_plausible_repair", None)
        assert callable(predicate), "guard.is_plausible_repair must be available"
        assert predicate(candidate, chunk) is True

    def test_unrelated_name_is_not_a_plausible_repair(self):
        predicate = getattr(guard, "is_plausible_repair", None)
        assert callable(predicate), "guard.is_plausible_repair must be available"
        assert predicate("Rockefeller", "A large crowd gathered outside.") is False


class TestRequestedGuardRegressions:
    def test_changed_14_to_11_is_rejected_and_original_kept(self):
        original = "The notice was printed on page 14 of the paper."
        corrected = "The notice was printed on page 11 of the paper."

        result = validate_chunk(original, corrected)

        assert not result.accepted
        assert result.final_text == original
        assert any("digit_changed" in reason for reason in result.flag_reasons)

    def test_ocr_number_repair_is_rejected(self):
        original = "The date on the document was I9Z6."
        corrected = "The date on the document was 1926."

        result = validate_chunk(original, corrected)

        assert not result.accepted
        assert result.final_text == original
        assert any("digit_changed" in reason for reason in result.flag_reasons)

    def test_heavy_rewrite_is_reverted(self):
        original = "The committee met in the hall and discussed the proposal."
        corrected = "A storm crossed the western counties before dawn."

        result = validate_chunk(original, corrected)

        assert result.similarity_ratio < 0.80
        assert not result.accepted
        assert result.final_text == original

    def test_empty_chunk_does_not_crash(self):
        result = validate_chunk("", "")

        assert result.accepted
        assert result.final_text == ""
