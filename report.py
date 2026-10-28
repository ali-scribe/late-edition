"""Report generator: renders static HTML report via Jinja2 with scan image, OCR, restored text, edits, and hauntings."""

import base64
import difflib
import html
import logging
from pathlib import Path
from typing import Any, Optional

import jinja2

from guard import GuardResult

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.jinja2"


def _encode_image_to_data_uri(image_path: Path) -> Optional[str]:
    """Encode an image to a base64 data URI for standalone HTML embedding."""
    if not image_path.exists() or not image_path.is_file():
        return None
    try:
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }
        mime = mime_types.get(suffix, "image/jpeg")
        data = image_path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception as exc:
        logger.warning(f"Could not encode image {image_path} to base64: {exc}")
        return None


def generate_html_diff(guard_result: GuardResult) -> str:
    """Generate inline HTML diff with highlighted edits for a chunk."""
    original = guard_result.original_text
    corrected = guard_result.corrected_text
    accepted = guard_result.accepted

    matcher = difflib.SequenceMatcher(None, original, corrected)
    html_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_slice = html.escape(original[i1:i2])
        corr_slice = html.escape(corrected[j1:j2])

        if tag == "equal":
            html_parts.append(orig_slice)
        elif tag == "replace":
            if accepted:
                html_parts.append(
                    f'<del class="diff-del" title="Original">{orig_slice}</del>'
                    f'<ins class="diff-ins" title="Restored">{corr_slice}</ins>'
                )
            else:
                html_parts.append(
                    f'<span class="diff-flagged" title="Rejected change">{orig_slice} '
                    f'<em>[Attempted: {corr_slice}]</em></span>'
                )
        elif tag == "delete":
            if accepted:
                html_parts.append(f'<del class="diff-del" title="Removed">{orig_slice}</del>')
            else:
                html_parts.append(f'<span class="diff-flagged" title="Attempted removal">{orig_slice}</span>')
        elif tag == "insert":
            if accepted:
                html_parts.append(f'<ins class="diff-ins" title="Inserted">{corr_slice}</ins>')
            else:
                html_parts.append(
                    f'<span class="diff-flagged" title="Rejected insertion">'
                    f'<em>[Rejected: {corr_slice}]</em></span>'
                )

    return "".join(html_parts)


def generate_report(
    newspaper_title: str,
    date: str,
    lccn: str,
    sequence: int,
    image_path: Path,
    raw_ocr_text: str,
    guard_results: list[GuardResult],
    output_path: Path = Path("report.html"),
    model_name: str = "gemma-4",
) -> Path:
    """Produce static HTML report using Jinja2 template."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Base64 encode image for self-contained portability
    image_data_uri = _encode_image_to_data_uri(image_path)
    image_rel_path = str(image_path)

    # Restored full text
    restored_text = "\n\n".join(r.final_text for r in guard_results)

    # Compute statistics
    total_chunks = len(guard_results)
    accepted_chunks = sum(1 for r in guard_results if r.accepted)
    flagged_chunks = total_chunks - accepted_chunks
    avg_similarity = (
        sum(r.similarity_ratio for r in guard_results) / total_chunks
        if total_chunks > 0
        else 1.0
    )

    # Collect flagged hauntings / hallucinations
    hauntings_list = []
    chunk_analyses = []

    for idx, r in enumerate(guard_results, start=1):
        # Generate inline diff HTML
        diff_html = generate_html_diff(r)
        chunk_analyses.append({
            "accepted": r.accepted,
            "similarity_ratio": r.similarity_ratio,
            "flag_reasons": r.flag_reasons,
            "html_diff": diff_html,
        })

        if not r.accepted:
            orig_snip = (r.original_text[:140] + "...") if len(r.original_text) > 140 else r.original_text
            corr_snip = (r.corrected_text[:140] + "...") if len(r.corrected_text) > 140 else r.corrected_text
            hauntings_list.append({
                "chunk_index": idx,
                "reason": "; ".join(r.flag_reasons),
                "hauntings": r.hauntings,
                "original_snippet": orig_snip,
                "attempted_snippet": corr_snip,
            })

    total_hauntings = sum(len(r.hauntings) for r in guard_results)

    # Render Jinja2 template
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_PATH.parent),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_PATH.name)

    rendered_html = template.render(
        newspaper_title=newspaper_title,
        date=date,
        lccn=lccn,
        sequence=sequence,
        model_name=model_name,
        image_data_uri=image_data_uri,
        image_rel_path=image_rel_path,
        raw_ocr_text=raw_ocr_text,
        restored_text=restored_text,
        total_chunks=total_chunks,
        accepted_chunks=accepted_chunks,
        flagged_chunks=flagged_chunks,
        total_hauntings=total_hauntings,
        avg_similarity=avg_similarity,
        hauntings=hauntings_list,
        chunk_analyses=chunk_analyses,
    )

    output_path.write_text(rendered_html, encoding="utf-8")
    logger.info(f"Report successfully written to {output_path.resolve()}")
    return output_path
