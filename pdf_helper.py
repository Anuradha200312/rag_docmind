import io
import logging
from dataclasses import dataclass
import tiktoken
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Load tokeniser once
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning("Failed to load tiktoken tokenizer: %s. Falling back to simple encoder.", e)
    _tokenizer = None

@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str

def extract_pages(file_bytes: bytes) -> list[PageContent]:
    """
    Extract text from every page of a PDF, returning a list of PageContent objects.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[PageContent] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        # Clean whitespace but keep paragraph structure
        cleaned = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
        if cleaned:
            pages.append(PageContent(page_number=idx, text=cleaned))

    if not pages:
        raise ValueError(
            "This PDF contains no extractable text. "
            "It may be scanned or image-only."
        )

    return pages

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    if _tokenizer:
        try:
            return len(_tokenizer.encode(text, disallowed_special=()))
        except Exception as e:
            logger.warning("Token counting failed: %s. Falling back to word estimate.", e)
    return len(text.split())

def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = 4000,
    chunk_overlap: int = 800,
) -> list[dict]:
    """
    Split page texts into overlapping chunks with metadata.
    Returns a list of dicts with: page_number, chunk_index, chunk_text
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    global_idx = 0

    for page in pages:
        if not page.text.strip():
            continue
        page_chunks = splitter.split_text(page.text)
        for text in page_chunks:
            stripped = text.strip()
            if not stripped:
                continue
            chunks.append({
                "page_number": page.page_number,
                "chunk_index": global_idx,
                "chunk_text": stripped,
            })
            global_idx += 1

    return chunks
