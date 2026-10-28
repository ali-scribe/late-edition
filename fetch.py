"""Chronicling America (LOC) fetcher, rate limiter, cache manager, and offline loader."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import requests

from config import CACHE_DIR, LOC_RATE_LIMIT_PER_MINUTE

logger = logging.getLogger(__name__)

LOC_BASE_URL = "https://chroniclingamerica.loc.gov"


class RateLimiter:
    """Sliding-window rate limiter ensuring no more than max_requests per window_seconds."""

    def __init__(self, max_requests: int = 20, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_timestamps: list[float] = []

    def acquire(self) -> None:
        """Wait if necessary to stay within the rate limit before proceeding."""
        now = time.monotonic()
        # Remove timestamps older than window
        self.request_timestamps = [
            ts for ts in self.request_timestamps if now - ts < self.window_seconds
        ]
        if len(self.request_timestamps) >= self.max_requests:
            oldest = self.request_timestamps[0]
            sleep_time = self.window_seconds - (now - oldest) + 0.05
            if sleep_time > 0:
                logger.info(f"Rate limit reached ({self.max_requests}/min). Sleeping for {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            now = time.monotonic()
            self.request_timestamps = [
                ts for ts in self.request_timestamps if now - ts < self.window_seconds
            ]
        self.request_timestamps.append(time.monotonic())


# Global rate limiter instance for Library of Congress calls
loc_rate_limiter = RateLimiter(max_requests=LOC_RATE_LIMIT_PER_MINUTE, window_seconds=60.0)


@dataclass
class FetchResult:
    """Container for newspaper page metadata, raw OCR text, and image path."""
    date: str
    newspaper_title: str
    lccn: str
    sequence: int
    image_path: Path
    ocr_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    is_offline: bool = False


def _safe_slug(text: str) -> str:
    """Sanitize strings for filesystem cache keys."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text).strip("_")


def fetch_offline(folder: Path | str) -> FetchResult:
    """Load a newspaper page image and raw OCR .txt file from a local directory."""
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Offline directory does not exist or is not a directory: {folder_path}")

    # Search for OCR text file
    txt_files = list(folder_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt OCR file found in offline folder: {folder_path}")
    ocr_file = txt_files[0]
    ocr_text = ocr_file.read_text(encoding="utf-8", errors="replace")

    # Search for page image
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"}
    img_files = [p for p in folder_path.iterdir() if p.suffix.lower() in image_extensions]
    if not img_files:
        raise FileNotFoundError(f"No image file found in offline folder: {folder_path}")
    img_file = img_files[0]

    # Check for optional metadata json
    meta_files = list(folder_path.glob("*.json"))
    metadata = {}
    if meta_files:
        try:
            metadata = json.loads(meta_files[0].read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to parse metadata JSON {meta_files[0]}: {e}")

    date_str = metadata.get("date", folder_path.name)
    title_str = metadata.get("title", folder_path.name)
    lccn_str = metadata.get("lccn", "local-offline")
    seq = metadata.get("sequence", 1)

    return FetchResult(
        date=date_str,
        newspaper_title=title_str,
        lccn=lccn_str,
        sequence=seq,
        image_path=img_file,
        ocr_text=ocr_text,
        metadata=metadata,
        is_offline=True,
    )


def fetch_loc_page(
    date: str,
    newspaper: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> FetchResult:
    """Fetch a public-domain newspaper page from Chronicling America (Library of Congress).
    
    Adheres strictly to the max 20 requests/minute rate limit and caches every file in cache/.
    Cached files are never refetched.
    If the API fails (network or server error), raises an informative exception recommending
    offline mode.
    """
    effective_cache_dir = cache_dir or CACHE_DIR
    effective_cache_dir.mkdir(parents=True, exist_ok=True)

    # Normalize date
    clean_date = date.strip()
    lccn_part = _safe_slug(newspaper) if newspaper else "any"
    cache_prefix = f"loc_{clean_date}_{lccn_part}"

    # Check cache first: if image, OCR text, and meta exist, NEVER refetch!
    cached_ocr_files = list(effective_cache_dir.glob(f"{cache_prefix}*_ocr.txt"))
    cached_img_files = [
        f for f in effective_cache_dir.glob(f"{cache_prefix}*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    cached_meta_files = list(effective_cache_dir.glob(f"{cache_prefix}*_meta.json"))

    if cached_ocr_files and cached_img_files:
        logger.info(f"Loading cached newspaper page from {effective_cache_dir}")
        ocr_text = cached_ocr_files[0].read_text(encoding="utf-8", errors="replace")
        img_path = cached_img_files[0]
        meta = {}
        if cached_meta_files:
            try:
                meta = json.loads(cached_meta_files[0].read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return FetchResult(
            date=meta.get("date", clean_date),
            newspaper_title=meta.get("title", "Historic Newspaper (Cached)"),
            lccn=meta.get("lccn", newspaper or "unknown"),
            sequence=meta.get("sequence", 1),
            image_path=img_path,
            ocr_text=ocr_text,
            metadata=meta,
            is_offline=False,
        )

    # Perform API query to Chronicling America
    session = requests.Session()
    session.headers.update({"User-Agent": "late-edition/1.0 (Public Domain OCR Research Tool)"})

    search_url = f"{LOC_BASE_URL}/search/pages/results/"
    params = {
        "dateFilterType": "exact",
        "date1": clean_date,
        "format": "json",
    }
    if newspaper:
        params["lccn"] = newspaper.strip()

    try:
        loc_rate_limiter.acquire()
        resp = session.get(search_url, params=params, timeout=20)
        resp.raise_for_status()
        search_data = resp.json()
    except Exception as exc:
        raise ConnectionError(
            f"Failed to query Chronicling America API for date '{clean_date}': {exc}. "
            "Please check network connection or use manual mode with '--offline <folder>'."
        ) from exc

    items = search_data.get("items", [])
    if not items:
        raise ValueError(
            f"No newspaper pages found in Chronicling America for date '{clean_date}'"
            + (f" and newspaper '{newspaper}'." if newspaper else ".")
            + " Check date format (YYYY-MM-DD) or use offline mode with '--offline <folder>'."
        )

    page_item = items[0]
    page_seq = page_item.get("sequence", 1)
    lccn_id = page_item.get("lccn", newspaper or "unknown")
    newspaper_title = page_item.get("title", "Historic Newspaper")
    page_json_url = page_item.get("url")

    # If page JSON url is provided, get full details
    ocr_url = None
    image_url = None

    if page_json_url:
        try:
            loc_rate_limiter.acquire()
            page_resp = session.get(page_json_url, timeout=20)
            page_resp.raise_for_status()
            page_details = page_resp.json()
            ocr_url = page_details.get("text") or page_details.get("ocr")
            # Image candidates
            image_url = (
                page_details.get("image")
                or page_details.get("thumbnail")
            )
        except Exception as exc:
            logger.warning(f"Could not load page JSON {page_json_url}: {exc}")

    # Fallback to direct OCR and image URLs based on page item id if needed
    item_id = page_item.get("id", "").strip("/")
    if not ocr_url and item_id:
        ocr_url = f"{LOC_BASE_URL}/{item_id}/ocr.txt"
    if not image_url and item_id:
        image_url = f"{LOC_BASE_URL}/{item_id}/image_683x1024.jpg"

    # Fetch OCR text
    ocr_text = ""
    if ocr_url:
        try:
            loc_rate_limiter.acquire()
            ocr_resp = session.get(ocr_url, timeout=20)
            ocr_resp.raise_for_status()
            ocr_text = ocr_resp.text
        except Exception as exc:
            # Fall back to ocr_eng embedded in search item if available
            ocr_text = page_item.get("ocr_eng", "")
            if not ocr_text:
                raise ConnectionError(f"Failed to fetch OCR text from {ocr_url}: {exc}") from exc
    elif "ocr_eng" in page_item:
        ocr_text = page_item["ocr_eng"]
    else:
        raise ValueError("Could not find OCR text in Chronicling America response.")

    # Fetch Page Image
    page_key = f"{cache_prefix}_seq{page_seq}"
    img_dest = effective_cache_dir / f"{page_key}_image.jpg"
    if image_url:
        try:
            loc_rate_limiter.acquire()
            img_resp = session.get(image_url, timeout=30)
            img_resp.raise_for_status()
            img_dest.write_bytes(img_resp.content)
        except Exception as exc:
            logger.warning(f"Failed to download primary image from {image_url}: {exc}")
            # Try thumbnail if larger image failed
            thumb_url = f"{LOC_BASE_URL}/{item_id}/thumbnail.jpg"
            loc_rate_limiter.acquire()
            thumb_resp = session.get(thumb_url, timeout=20)
            thumb_resp.raise_for_status()
            img_dest.write_bytes(thumb_resp.content)
    else:
        raise ValueError("Could not locate page image URL in Chronicling America metadata.")

    # Save OCR text and metadata to cache
    ocr_dest = effective_cache_dir / f"{page_key}_ocr.txt"
    ocr_dest.write_text(ocr_text, encoding="utf-8")

    meta_dict = {
        "date": clean_date,
        "title": newspaper_title,
        "lccn": lccn_id,
        "sequence": page_seq,
        "page_url": page_json_url,
        "ocr_url": ocr_url,
        "image_url": image_url,
    }
    meta_dest = effective_cache_dir / f"{page_key}_meta.json"
    meta_dest.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    return FetchResult(
        date=clean_date,
        newspaper_title=newspaper_title,
        lccn=lccn_id,
        sequence=page_seq,
        image_path=img_dest,
        ocr_text=ocr_text,
        metadata=meta_dict,
        is_offline=False,
    )
