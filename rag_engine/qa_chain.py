"""
QA Chain Module
Builds the retrieval-augmented generation chain supporting multiple LLM providers:
  • Groq (Free & Blazing Fast: Llama 3.3 70B, Llama 3.1 8B)
  • OpenAI (GPT-4o, GPT-4o-mini)
  • OpenRouter (Llama 3.3, DeepSeek R1, Free models)
  • Google Gemini (Gemini 1.5 Flash, 2.0 Flash)
  • Ollama / Local (Run offline models locally without keys)
"""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_RAG = """\
You are a knowledgeable, precise, and helpful AI assistant. \
Your task is to answer the user's question based **only** on the provided context \
extracted from their uploaded PDF documents.

Rules:
1. Answer accurately using ONLY the information in the context below.
2. If the context does not contain enough information to answer, say: \
   "I don't have enough information in the uploaded documents to answer this question."
3. When relevant, cite the source document and page number.
4. Be concise but thorough. Use bullet points or numbered lists when appropriate.
5. Do NOT make up information or use external knowledge.
"""

SYSTEM_PROMPT_FULL = """\
You are a knowledgeable, precise, and helpful AI assistant. \
You have been given the **complete text** of the user's uploaded PDF document(s). \
Answer the user's question based **only** on this text.

Rules:
1. You have the FULL document — use all of it to give a comprehensive answer.
2. If the question is unrelated to the document content, say so.
3. When relevant, cite the page number.
4. Be concise but thorough. Use bullet points or numbered lists when appropriate.
5. Do NOT make up information or use external knowledge.
"""

QA_PROMPT_TEMPLATE = """\
Context from uploaded documents:
---
{context}
---

User Question: {question}

Answer:"""


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        header = f"[Source: {source} | Page: {page} | Chunk: {chunk_idx}]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)


class QAChain:
    """Multi-provider RAG Question-Answering chain."""

    def __init__(
        self,
        provider: str = "groq",
        api_key: str = "",
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        base_url: Optional[str] = None,
    ):
        provider = provider.lower().strip()
        self.provider = provider
        self.model_name = model_name

        key = api_key.strip()

        if provider == "groq":
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model=model_name,
                groq_api_key=key,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=key,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )

        elif provider == "openrouter":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=key,
                base_url=base_url or "https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_output_tokens,
            )

        elif provider == "ollama":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model_name or "llama3.2",
                api_key="ollama",
                base_url=base_url or "http://localhost:11434/v1",
                temperature=temperature,
            )

        elif provider in ("gemini", "google"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self.output_parser = StrOutputParser()

        # Build prompt chains
        self.prompt_rag = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT_RAG), ("human", QA_PROMPT_TEMPLATE)]
        )
        self.prompt_full = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT_FULL), ("human", QA_PROMPT_TEMPLATE)]
        )

        self._chain_rag = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | self.prompt_rag
            | self.llm
            | self.output_parser
        )
        self._chain_full = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | self.prompt_full
            | self.llm
            | self.output_parser
        )

    def answer(
        self,
        question: str,
        retrieved_docs: List[Document],
        full_context: bool = False,
    ) -> dict:
        """
        Generate an answer given a question and retrieved documents.
        """
        context_str = format_docs(retrieved_docs)

        chain = self._chain_full if full_context else self._chain_rag
        answer_text = chain.invoke(
            {"context": context_str, "question": question}
        )

        # Collect source info
        sources = []
        for doc in retrieved_docs:
            sources.append(
                {
                    "file": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "chunk": doc.metadata.get("chunk_index", "?"),
                    "preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                }
            )

        return {
            "answer": answer_text,
            "sources": sources,
            "context": context_str,
            "mode": "full-context" if full_context else "rag-retrieval",
            "provider": self.provider,
            "model": self.model_name,
        }
