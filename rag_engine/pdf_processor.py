"""
PDF Processing Module
Handles PDF loading, text extraction, and intelligent chunking.
Optimized for small-to-medium documents (~2000 words / ~10K chars).
"""

import os
import hashlib
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ~2000 words ≈ 10,000–12,000 characters.
# With 500-char chunks + 100 overlap we get ~20–25 fine-grained chunks,
# which means top-K retrieval covers more of the document precisely.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# If the entire document text is below this threshold (chars) we flag it
# as "small" so the QA chain can send ALL chunks instead of only top-K.
SMALL_DOC_THRESHOLD = 15_000  # ~2500 words


class PDFProcessor:
    """Processes PDF files: extracts text and splits into chunks for RAG."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate SHA-256 hash of a file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF and return a list of Documents (one per page)."""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return pages

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for embedding."""
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    @staticmethod
    def compute_stats(pages: List[Document]) -> dict:
        """Compute word count, char count, and page count for the document."""
        full_text = " ".join(p.page_content for p in pages)
        word_count = len(full_text.split())
        char_count = len(full_text)
        return {
            "pages": len(pages),
            "words": word_count,
            "chars": char_count,
            "is_small": char_count <= SMALL_DOC_THRESHOLD,
        }

    def process_pdf(
        self, file_path: str, source_name: str | None = None
    ) -> Tuple[List[Document], str, dict]:
        """
        Full pipeline: load PDF → chunk → return (chunks, file_hash, stats).
        Adds metadata (source filename, file hash, chunk index, doc_is_small)
        to each chunk.
        """
        file_hash = self.get_file_hash(file_path)
        pages = self.load_pdf(file_path)

        # Compute stats
        stats = self.compute_stats(pages)

        # Add source metadata
        source = source_name or os.path.basename(file_path)
        for page in pages:
            page.metadata["source_file"] = source
            page.metadata["file_hash"] = file_hash

        chunks = self.chunk_documents(pages)

        # Add chunk index + small-doc flag
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["doc_is_small"] = stats["is_small"]

        stats["chunk_count"] = len(chunks)
        return chunks, file_hash, stats
