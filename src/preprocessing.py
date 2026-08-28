"""Paper-style preprocessing for Kannada-English social-media comments.

The HASTIKA paper names the preprocessing stages but does not publish its
Kannada stopword list or lemmatizer. This module therefore keeps those
approximations explicit and versioned instead of silently inventing behavior.
"""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import ftfy
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import PreprocessConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_DIR = _PROJECT_ROOT / "resources"

_BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\S+")
_HASHTAG_RE = re.compile(r"(?<!\w)#\S+")
_ASCII_WORD_RE = re.compile(r"^[a-z]+$")
_WHITESPACE_RE = re.compile(r"\s+")

# These words are explicitly contextual in the paper or important negations.
# They are removed from every stopword source before preprocessing begins.
_PROTECTED_WORDS = {
    "sir",
    "devru",
    "kannadiga",
    "namma",
    "nadu",
    "avanu",
    "avalu",
    "dharma",
    "no",
    "not",
    "nor",
    "never",
    "alla",
    "illa",
    "beda",
}

_NLTK_RESOURCES = (
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)


def _nltk_resource_exists(locator: str) -> bool:
    """Return whether an NLTK resource exists unpacked or as a ZIP archive."""

    for candidate in (locator, f"{locator}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_resources(download_missing: bool = True) -> None:
    """Ensure every language resource required by preprocessing is available.

    Kaggle must have Internet enabled on the first run. Missing resources are
    never ignored because doing so would silently change the model inputs.
    """

    missing: list[tuple[str, str]] = []
    for locator, package_name in _NLTK_RESOURCES:
        if not _nltk_resource_exists(locator):
            missing.append((locator, package_name))

    if download_missing:
        for locator, package_name in missing:
            try:
                downloaded = nltk.download(package_name, quiet=True)
            except Exception as error:  # pragma: no cover - network dependent.
                raise RuntimeError(
                    f"Failed to download NLTK resource '{package_name}'. Enable "
                    "Internet in Kaggle and run the script again."
                ) from error
            if not downloaded or not _nltk_resource_exists(locator):
                raise RuntimeError(
                    f"NLTK resource '{package_name}' is unavailable. Enable "
                    "Internet in Kaggle and run the script again."
                )

    still_missing = [
        package_name
        for locator, package_name in _NLTK_RESOURCES
        if not _nltk_resource_exists(locator)
    ]
    if still_missing:
        raise RuntimeError(
            "Missing NLTK resources: "
            f"{', '.join(still_missing)}. Enable Internet in Kaggle and rerun."
        )


def _read_stopword_file(filename: str) -> set[str]:
    """Read a UTF-8 stopword file while ignoring comments and blank lines."""

    path = _RESOURCE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required stopword resource was not found: {path}")
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


@lru_cache(maxsize=1)
def _stopword_set() -> frozenset[str]:
    """Build the reproducible English, Kannada and romanized stopword set."""

    ensure_nltk_resources(download_missing=False)
    words = set(stopwords.words("english"))
    words.update(_read_stopword_file("kannada_stopwords.txt"))
    words.update(_read_stopword_file("romanized_kannada_stopwords.txt"))
    words.difference_update(_PROTECTED_WORDS)
    return frozenset(words)


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    """Return the shared English WordNet lemmatizer."""

    ensure_nltk_resources(download_missing=False)
    return WordNetLemmatizer()


def _keep_letters_marks_numbers_and_spaces(text: str) -> str:
    """Remove punctuation/symbols without breaking Kannada combining marks."""

    cleaned = []
    for character in text:
        category_group = unicodedata.category(character)[0]
        cleaned.append(
            character
            if character.isspace() or category_group in {"L", "M", "N"}
            else " "
        )
    return "".join(cleaned)


def _wordnet_pos(penn_tag: str) -> str:
    """Map a Penn Treebank tag to the closest WordNet part of speech."""

    if penn_tag.startswith("J"):
        return "a"
    if penn_tag.startswith("V"):
        return "v"
    if penn_tag.startswith("R"):
        return "r"
    return "n"


def _lemmatize_english_tokens(tokens: list[str]) -> list[str]:
    """POS-tag and lemmatize ASCII English tokens while preserving all others."""

    english_positions = [
        index for index, token in enumerate(tokens) if _ASCII_WORD_RE.fullmatch(token)
    ]
    if not english_positions:
        return tokens

    english_tokens = [tokens[index] for index in english_positions]
    tagged_tokens = nltk.pos_tag(english_tokens, lang="eng")
    lemmatizer = _lemmatizer()
    output = list(tokens)
    for index, (token, tag) in zip(english_positions, tagged_tokens, strict=True):
        output[index] = lemmatizer.lemmatize(token, pos=_wordnet_pos(tag))
    return output


def _normalize_repeated_characters(tokens: list[str], max_repeats: int) -> list[str]:
    """Limit a run of one character to ``max_repeats`` occurrences."""

    repeat_limit = max(1, max_repeats)
    repeated_re = re.compile(r"(.)\1{" + str(repeat_limit) + r",}", re.DOTALL)
    return [
        repeated_re.sub(lambda match: match.group(1) * repeat_limit, token)
        for token in tokens
    ]


def _collapse_repeated_words(tokens: list[str]) -> list[str]:
    """Collapse immediately repeated emphasis tokens to one occurrence."""

    output: list[str] = []
    for token in tokens:
        if not output or output[-1] != token:
            output.append(token)
    return output


def normalize_text(text: object, config: PreprocessConfig) -> str:
    """Apply the published HASTIKA stages in their documented order."""

    value = "" if pd.isna(text) else str(text)
    if config.fix_unicode:
        value = ftfy.fix_text(value)
        value = unicodedata.normalize("NFKC", value)
    if config.decode_html_entities:
        value = html.unescape(value)
    if config.lowercase:
        value = value.lower()

    if config.remove_html:
        value = _BR_TAG_RE.sub(" ", value)
        value = _HTML_TAG_RE.sub(" ", value)
    if config.remove_urls:
        value = _URL_RE.sub(" ", value)
    if config.remove_mentions:
        value = _MENTION_RE.sub(" ", value)
    if config.remove_hashtags:
        value = _HASHTAG_RE.sub(" ", value)
    if config.remove_non_alphanumeric:
        value = _keep_letters_marks_numbers_and_spaces(value)

    # Noise removal leaves whitespace-delimited Unicode word tokens.
    tokens = _WHITESPACE_RE.sub(" ", value).strip().split()
    if config.remove_stopwords:
        blocked = _stopword_set()
        tokens = [token for token in tokens if token not in blocked]
    if config.lemmatize_english:
        tokens = _lemmatize_english_tokens(tokens)
    if config.normalize_repeated_characters:
        tokens = _normalize_repeated_characters(
            tokens, config.max_repeated_characters
        )
    if config.normalize_repeated_words:
        tokens = _collapse_repeated_words(tokens)
    return " ".join(tokens)
