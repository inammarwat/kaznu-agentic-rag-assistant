import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import requests
import trafilatura
import urllib3
from bs4 import BeautifulSoup

from kaznu_rag.schemas import DocumentRecord
from kaznu_rag.preprocess.text_cleaning import clean_text
from kaznu_rag.utils import sha256_text

logger = logging.getLogger(__name__)

# Suppress warnings only for the explicit SSL fallback path.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def read_urls(urls_file: Path) -> List[str]:
    """
    Read valid URLs from urls.txt.

    Ignores:
    - blank lines
    - comment/header lines beginning with '#'
    """
    urls: List[str] = []

    with urls_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("http://") or line.startswith("https://"):
                urls.append(line)

    return urls


def infer_web_group(url: str) -> str:
    """Infer source group from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    if "welcome.kaznu.kz" in domain:
        return "foreign_applicant_domain"

    if "farabi.university" in domain and "/students" in path:
        return "student_domain"

    if "farabi.university" in domain or "welcome.kaznu.kz" in domain:
        return "university_intro"

    return "unknown_web"


def safe_filename_from_url(url: str) -> str:
    """Convert URL into filesystem-safe HTML filename."""
    value = re.sub(r"[^a-zA-Z0-9]+", "_", url)
    return value.strip("_")[:180] + ".html"


def build_headers(user_agent: str) -> dict:
    """
    Browser-like headers.

    KazNU pages are more reliably fetched when requests look like
    a normal browser request rather than a minimal Python client.
    """
    return {
        "User-Agent": user_agent
        or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8,kk;q=0.7",
        "Connection": "keep-alive",
    }


def fetch_html(url: str, timeout_seconds: int, user_agent: str) -> Optional[str]:
    """
    Fetch a web page.

    Strategy:
    1. Try normal SSL verification first.
    2. If SSL certificate verification fails, retry with verify=False.
       This matches the successful behavior from your previous notebook.
    """
    headers = build_headers(user_agent)

    # First attempt: standard verified HTTPS.
    try:
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers=headers,
            verify=True,
        )
        response.raise_for_status()
        logger.info("Fetched with SSL verification: %s | status=%s", url, response.status_code)
        return response.text

    except requests.exceptions.SSLError as exc:
        logger.warning(
            "SSL verification failed. Retrying with verify=False: %s | error=%s",
            url,
            exc,
        )

        # Second attempt: explicit SSL fallback.
        try:
            response = requests.get(
                url,
                timeout=timeout_seconds,
                headers=headers,
                verify=False,
            )
            response.raise_for_status()
            logger.info("Fetched with SSL fallback: %s | status=%s", url, response.status_code)
            return response.text

        except requests.RequestException as fallback_exc:
            logger.error(
                "Failed after SSL fallback: %s | error=%s",
                url,
                fallback_exc,
            )
            return None

    except requests.RequestException as exc:
        logger.error("Failed to fetch URL: %s | error=%s", url, exc)
        return None


def extract_title(html: str) -> Optional[str]:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    if soup.title and soup.title.text:
        return soup.title.text.strip()

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)

    return None


def extract_main_text(html: str, url: str) -> str:
    """
    Extract readable text from HTML.

    First uses trafilatura. If extraction is weak or empty,
    falls back to BeautifulSoup.
    """
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_formatting=False,
        favor_recall=True,
    )

    if extracted and len(extracted.strip()) > 100:
        return extracted

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    return soup.get_text("\n", strip=True)


def extract_web_records(
    urls_file: Path,
    html_output_dir: Path,
    timeout_seconds: int,
    user_agent: str,
    min_text_chars: int = 100,
) -> List[DocumentRecord]:
    """
    Extract web pages into normalized DocumentRecord objects.
    """
    records: List[DocumentRecord] = []
    urls = read_urls(urls_file)

    html_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Found %s URLs", len(urls))

    for url in urls:
        logger.info("Fetching URL: %s", url)

        html = fetch_html(
            url=url,
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
        )

        if not html:
            logger.warning("No HTML returned for URL: %s", url)
            continue

        html_path = html_output_dir / safe_filename_from_url(url)
        html_path.write_text(html, encoding="utf-8")

        title = extract_title(html)
        raw_text = extract_main_text(html, url=url)
        text = clean_text(raw_text)

        if len(text) < min_text_chars:
            logger.warning(
                "Skipping short web page after extraction: url=%s chars=%s",
                url,
                len(text),
            )
            continue

        checksum = sha256_text(text)
        parsed = urlparse(url)
        collected_at = datetime.now(timezone.utc).isoformat()
        document_group = infer_web_group(url)

        doc_id = f"web::{parsed.netloc}{parsed.path}::{checksum[:12]}"

        records.append(
            DocumentRecord(
                doc_id=doc_id,
                source_type="web",
                source_name=url,
                content_type="web_page",
                text=text,
                url=url,
                title=title,
                metadata={
                    "document_group": document_group,
                    "authority_level": "official_web",
                    "domain": parsed.netloc,
                    "path": parsed.path,
                    "query": parsed.query,
                    "raw_html_path": str(html_path),
                    "collected_at": collected_at,
                    "checksum": checksum,
                    "token_estimate": max(1, len(text.split())),
                    "ssl_strategy": "verify_true_then_verify_false_fallback",
                },
            )
        )

        logger.info(
            "Web record created: url=%s chars=%s title=%s",
            url,
            len(text),
            title,
        )

    return records