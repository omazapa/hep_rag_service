#!/usr/bin/env python3
"""
LangChain-based HEP Documentation Indexer

This module provides a LangChain wrapper for indexing and searching
HEP documentation using Elasticsearch and embeddings.
"""

from typing import Any

from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class LangChainHEPIndexer:
    """
    LangChain-based indexer for HEP documentation.

    This class provides a high-level interface for indexing and searching
    HEP documentation using LangChain's ElasticsearchStore with hybrid search
    capabilities (BM25 + vector similarity).

    Attributes:
        embeddings: HuggingFace embeddings model
        vector_store: Elasticsearch vector store
        text_splitter: Text splitter for chunking documents
    """

    def __init__(
        self,
        es_host: str = "http://localhost:9200",
        index_name: str = "hep-docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the LangChain HEP indexer.

        Args:
            es_host: Elasticsearch host URL
            index_name: Name of the Elasticsearch index
            embedding_model: HuggingFace model name for embeddings
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
        )

        # Initialize Elasticsearch vector store with approximate retrieval (vector-only)
        self.vector_store = ElasticsearchStore(
            es_url=es_host,
            index_name=index_name,
            embedding=self.embeddings,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
        )

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len,
        )

    def index_documents(self, docs: list[dict]) -> None:
        """
        Index a list of documents into Elasticsearch.

        Args:
            docs: List of document dictionaries with keys:
                - content: Document text content
                - title: Document title
                - url: Document URL
                - category: Document category (e.g., 'root', 'geant4')
                - type: Document type (e.g., 'html', 'pdf')
        """
        # Convert to LangChain Document format
        langchain_docs = [
            Document(
                page_content=doc["content"],
                metadata={
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "category": doc.get("category", ""),
                    "type": doc.get("type", ""),
                },
            )
            for doc in docs
        ]

        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(langchain_docs)

        # Add to vector store
        self.vector_store.add_documents(split_docs)

        print(f"✓ Indexed {len(split_docs)} chunks from {len(docs)} documents")

    def search(self, query: str, k: int = 5, filter_dict: dict[str, Any] | None = None) -> list[dict]:
        """
        Search for documents using hybrid search (BM25 + vector similarity).

        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"category": "root"})

        Returns:
            List of dictionaries containing:
                - content: Document content
                - title: Document title
                - url: Document URL
                - category: Document category
                - type: Document type
                - score: Relevance score
        """
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "category": doc.metadata.get("category", ""),
                    "type": doc.metadata.get("type", ""),
                    "score": float(score),
                }
            )

        return formatted_results

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None):
        """
        Get a LangChain retriever interface.

        This method returns a retriever that can be used in LangChain chains
        like RetrievalQA or ConversationalRetrievalChain.

        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 5})

        Returns:
            LangChain retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def delete_index(self) -> None:
        """Delete the Elasticsearch index."""
        self.vector_store.client.indices.delete(index=self.vector_store.index_name, ignore=[400, 404])
        print(f"✓ Deleted index: {self.vector_store.index_name}")
