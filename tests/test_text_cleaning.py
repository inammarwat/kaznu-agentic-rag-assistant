from kaznu_rag.preprocess.text_cleaning import clean_text


def test_clean_text_removes_extra_whitespace():
    raw = "This   is     a test.\n\n\nAnother line."
    cleaned = clean_text(raw)

    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_fixes_hyphenation():
    raw = "educa-\ntion"
    cleaned = clean_text(raw)

    assert "education" in cleaned