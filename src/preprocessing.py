"""Text cleaning functions for the RAG corpus."""
import html
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # langdetect is non-deterministic without a seed
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|br|li|ul|ol|tr|h[1-6]|table|blockquote)\b[^>]*>", re.IGNORECASE
)
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_QUOTE_MAP = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
)
_TRAILING_SECTION_RE = re.compile(
    r"^(?:References|Other websites|External links|Related pages|Further reading)[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


# 4.1 HTML stripping
def strip_html(text: str) -> str:
    """Remove HTML tags (keeping block breaks as newlines) and decode entities."""
    if not text:
        return ""
    if _TAG_RE.search(text):
        text = _BLOCK_TAG_RE.sub("\n", text)
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text()
    return html.unescape(text)


# 4.2 Unicode normalization
def normalize_unicode(text: str) -> str:
    """NFKC-normalize, drop zero-width/control chars, straighten curly quotes."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.translate(_QUOTE_MAP)
    return _CONTROL_RE.sub("", text)


# 4.3 Whitespace cleanup
def clean_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, trim lines, cap blank lines at one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 4.4 Language filtering
def detect_language(text: str, sample_chars: int = 1000) -> str:
    """Return an ISO language code for the text, or 'unknown'."""
    sample = text[:sample_chars].strip()
    if len(sample) < 20:
        return "unknown"
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def is_english(text: str) -> bool:
    return detect_language(text) == "en"


# Wikipedia-specific: drop link-list sections that add no retrievable content
def remove_trailing_sections(text: str) -> str:
    """Cut the text at the first References / Other websites / etc. heading."""
    match = _TRAILING_SECTION_RE.search(text)
    return text[: match.start()].rstrip() if match else text


# 4.5 Full pipeline
def clean_document(text: str) -> str:
    """Run all cleaning steps in order. Language filtering is applied separately."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    return remove_trailing_sections(text)