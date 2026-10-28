import re

from chunk import split_into_chunks


def _words(text):
    return re.findall(r"\b[\w'-]+\b", text)


def test_chunks_are_approximately_at_most_800_characters():
    sentences = [
        f"Sentence {index} describes the newspaper article and its historical details."
        for index in range(60)
    ]
    chunks = split_into_chunks(" ".join(sentences))

    assert chunks
    for chunk in chunks:
        if len(chunk) > 800:
            sentences_in_chunk = re.split(r"(?<=[.!?])\s+", chunk.strip())
            assert len(sentences_in_chunk) == 1
            assert len(sentences_in_chunk[0]) > 800


def test_chunk_boundaries_never_split_a_word():
    text = " ".join(
        f"Unusuallylongword{index} appears in this newspaper sentence."
        for index in range(80)
    )

    chunks = split_into_chunks(text)

    assert all(not chunk.startswith(" ") and not chunk.endswith(" ") for chunk in chunks)
    assert _words(" ".join(chunks)) == _words(text)


def test_joining_chunks_preserves_every_original_word_in_order():
    sentences = [
        f"Article {index} reports that the committee met in Albany on Monday."
        for index in range(60)
    ]
    original = " ".join(sentences)

    chunks = split_into_chunks(original)

    assert _words(" ".join(chunks)) == _words(original)
