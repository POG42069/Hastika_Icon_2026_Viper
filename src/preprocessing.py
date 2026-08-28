"""Conservative preprocessing for Kannada-English social-media comments."""

from __future__ import annotations

import html
import re
import unicodedata

import ftfy
import pandas as pd

from config import PreprocessConfig

_BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"#(?=\w)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: object, config: PreprocessConfig) -> str:
    """Normalize noise without discarding useful hate-speech cues.

    MuRIL is cased, so lowercasing is disabled by default. Emoji, punctuation,
    stopwords and inflectional information are retained because they may carry
    sentiment, emphasis, negation, or code-mixed context.
    """

    value = "" if pd.isna(text) else str(text)
    if config.fix_unicode:
        value = ftfy.fix_text(value)
        value = unicodedata.normalize("NFKC", value)
    if config.decode_html_entities:
        value = html.unescape(value)

    value = _BR_TAG_RE.sub(" ", value)
    value = _HTML_TAG_RE.sub(" ", value)
    if config.replace_urls:
        value = _URL_RE.sub(" URL ", value)
    if config.replace_mentions:
        value = _MENTION_RE.sub(" USER ", value)
    if config.keep_hashtag_text:
        value = _HASHTAG_RE.sub("", value)

    if config.normalize_repeated_characters:
        max_repeats = max(1, config.max_repeated_characters)
        repeated_re = re.compile(r"(.)\1{" + str(max_repeats) + r",}", flags=re.DOTALL)
        value = repeated_re.sub(lambda match: match.group(1) * max_repeats, value)

    if config.lowercase:
        value = value.lower()
    return _WHITESPACE_RE.sub(" ", value).strip()
