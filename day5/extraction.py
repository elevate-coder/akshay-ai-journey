"""
RAG Pipeline — Extraction Stage (pluggable)
=============================================
Registry of format-specific extractors. Add a new format by writing a
function `(path: str) -> str` and registering it in EXTRACTORS.
"""

from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
import docx


def extract_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def extract_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_html(path: str) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    # Drop script/style — their text isn't content and pollutes chunks/embeddings
    for tag in soup(["script", "style"]):
        tag.decompose()
    # get_text with a separator preserves rough block structure (headings,
    # paragraphs) so downstream chunking still has something to split on
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_txt,
    ".md": extract_txt,
    ".html": extract_html,
    ".htm": extract_html,
}


def extract(path: str, method: str | None = None) -> str:
    """
    Extract text from `path`. Auto-detects extractor from file extension
    unless `method` is given explicitly (e.g. to force .txt handling on a
    file with a nonstandard extension).
    """
    if method:
        if method not in EXTRACTORS:
            raise ValueError(f"Unknown extraction method '{method}'. Options: {list(EXTRACTORS)}")
        return EXTRACTORS[method](path)

    ext = Path(path).suffix.lower()
    if ext not in EXTRACTORS:
        raise ValueError(f"No extractor registered for '{ext}'. Options: {list(EXTRACTORS)}")
    return EXTRACTORS[ext](path)
