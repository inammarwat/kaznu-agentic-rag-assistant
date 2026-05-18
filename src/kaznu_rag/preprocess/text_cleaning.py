import re
import unicodedata


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_excess_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_broken_hyphenation(text: str) -> str:
    """
    Fix words split across lines, e.g.:
    educa-
    tion -> education
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def remove_common_pdf_noise(text: str) -> str:
    noise_patterns = [
        r"NJSC Al-Farabi Kazakh National University Date:.*",
        r"Academic Policy Page \d+ of \d+",
        r"FARABI UNIVERSITY",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = fix_broken_hyphenation(text)
    text = remove_common_pdf_noise(text)
    text = remove_excess_whitespace(text)
    return text