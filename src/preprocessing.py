"""Light normalization that preserves user-authored social-media content."""

from __future__ import annotations

import html
import re
import unicodedata

import ftfy
import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Repair encoding noise while preserving URLs, mentions, and HTML tags.

    Casing, hashtags, emoji, punctuation, stopwords, repeated characters, and
    repeated words are intentionally left intact for the LLM to interpret.
    """

    value = "" if pd.isna(text) else str(text)
    value = ftfy.fix_text(value)
    value = unicodedata.normalize("NFKC", value)
    value = html.unescape(value)
    return _WHITESPACE_RE.sub(" ", value).strip()
