"""
Vector Store Module
Manages ChromaDB vector store with HuggingFace embeddings.
Adds get_all_by_source() for full-context mode on small documents.
"""

import os
import shutil
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages ChromaDB vector store for document retrieval."""

    def __init__(
        self,
        persist_directory: str = "./vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "pdf_documents",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize embedding model (runs locally — no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize or load ChromaDB
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    @property
    def document_count(self) -> int:
        """Return the number of documents in the vector store."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def get_stored_hashes(self) -> set:
        """Get all unique file hashes already stored to avoid duplicates."""
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            hashes = set()
            for meta in results.get("metadatas", []):
                if meta and "file_hash" in meta:
                    hashes.add(meta["file_hash"])
            return hashes
        except Exception:
            return set()

    def get_stored_sources(self) -> List[str]:
        """Get list of unique source filenames in the store."""
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            sources = set()
            for meta in results.get("metadatas", []):
                if meta and "source_file" in meta:
                    sources.add(meta["source_file"])
            return sorted(sources)
        except Exception:
            return []

    def add_documents(self, documents: List[Document]) -> int:
        """Add document chunks to the vector store. Returns count added."""
        if not documents:
            return 0
        self.vectorstore.add_documents(documents)
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Document]:
        """
        Search for the most relevant chunks given a query.
        Optionally filter by source file.
        """
        filter_dict = None
        if source_filter:
            filter_dict = {"source_file": source_filter}

        results = self.vectorstore.similarity_search(
            query, k=k, filter=filter_dict
        )
        return results

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[tuple]:
        """Search and return (Document, score) tuples."""
        filter_dict = None
        if source_filter:
            filter_dict = {"source_file": source_filter}

        results = self.vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
        return results

    def get_all_by_source(self, source_file: str) -> List[Document]:
        """
        Return ALL chunks for a given source file, ordered by chunk_index.
        Used for small-document full-context mode.
        """
        try:
            results = self.vectorstore._collection.get(
                where={"source_file": source_file},
                include=["documents", "metadatas"],
            )
            docs = []
            for doc_text, meta in zip(
                results.get("documents", []), results.get("metadatas", [])
            ):
                docs.append(Document(page_content=doc_text, metadata=meta or {}))
            # Sort by chunk_index so context is in reading order
            docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
            return docs
        except Exception:
            return []

    def get_all_documents(self) -> List[Document]:
        """
        Return ALL chunks across all sources, ordered by source then chunk_index.
        Used when the total knowledge base is small enough for full context.
        """
        try:
            results = self.vectorstore._collection.get(
                include=["documents", "metadatas"]
            )
            docs = []
            for doc_text, meta in zip(
                results.get("documents", []), results.get("metadatas", [])
            ):
                docs.append(Document(page_content=doc_text, metadata=meta or {}))
            docs.sort(
                key=lambda d: (
                    d.metadata.get("source_file", ""),
                    d.metadata.get("chunk_index", 0),
                )
            )
            return docs
        except Exception:
            return []

    def is_small_corpus(self, threshold_chars: int = 15_000) -> bool:
        """Check if the total stored text is below the small-doc threshold."""
        try:
            results = self.vectorstore._collection.get(include=["documents"])
            total = sum(len(d) for d in results.get("documents", []))
            return total <= threshold_chars
        except Exception:
            return False

    def get_total_chars(self) -> int:
        """Return total character count of all stored chunks."""
        try:
            results = self.vectorstore._collection.get(include=["documents"])
            return sum(len(d) for d in results.get("documents", []))
        except Exception:
            return 0

    def get_total_words(self) -> int:
        """Return approximate total word count of all stored chunks."""
        try:
            results = self.vectorstore._collection.get(include=["documents"])
            return sum(len(d.split()) for d in results.get("documents", []))
        except Exception:
            return 0

    def delete_by_source(self, source_file: str) -> bool:
        """Delete all chunks belonging to a specific source file."""
        try:
            results = self.vectorstore._collection.get(
                where={"source_file": source_file}, include=["metadatas"]
            )
            ids = results.get("ids", [])
            if ids:
                self.vectorstore._collection.delete(ids=ids)
            return True
        except Exception:
            return False

    def clear_all(self) -> bool:
        """Delete the entire vector store and reinitialize."""
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            # Reinitialize
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            return True
        except Exception:
            return False
