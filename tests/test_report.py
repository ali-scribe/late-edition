import re

from guard import validate_chunk
from report import generate_report


def test_report_is_created_and_contains_flagged_reason(tmp_path):
    original = "The record was made in 1918."
    result = validate_chunk(original, "The record was made in 1919.")
    output = tmp_path / "report.html"

    written_path = generate_report(
        newspaper_title="The Example",
        date="1918-01-01",
        lccn="example",
        sequence=1,
        image_path=tmp_path / "missing-image.png",
        raw_ocr_text=original,
        guard_results=[result],
        output_path=output,
    )

    assert written_path == output
    assert output.is_file()
    assert "digit_changed" in output.read_text(encoding="utf-8")


def test_report_has_no_remote_src_or_href_attributes(tmp_path):
    result = validate_chunk("A clean sentence.", "A clean sentence.")
    output = tmp_path / "report.html"

    generate_report(
        newspaper_title="The Example",
        date="1918-01-01",
        lccn="example",
        sequence=1,
        image_path=tmp_path / "missing-image.png",
        raw_ocr_text="A clean sentence.",
        guard_results=[result],
        output_path=output,
    )
    rendered = output.read_text(encoding="utf-8")
    attributes = re.findall(r"\b(?:src|href)\s*=\s*['\"]([^'\"]*)['\"]", rendered, re.I)

    assert all(not value.lower().startswith(("http://", "https://")) for value in attributes)
