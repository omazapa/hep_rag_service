#!/usr/bin/env python3
"""
RAG System for Geant4 Documentation - Elasticsearch Indexing
Processes Sphinx HTML documentation and indexes in Elasticsearch with vector embeddings
"""

import hashlib
import logging
import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Generator, List

from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Chunking utilities
def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace"""
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If not the last chunk, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings (., !, ?, \n) within last 100 chars
            search_start = max(start, end - 100)
            last_period = text.rfind(".", search_start, end)
            last_newline = text.rfind("\n", search_start, end)
            last_question = text.rfind("?", search_start, end)
            last_exclamation = text.rfind("!", search_start, end)

            # Find the best breaking point
            break_point = max(last_period, last_newline, last_question, last_exclamation)
            if break_point > start:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap if end < len(text) else end

    return chunks


# Standalone function for parallel HTML extraction
def extract_html_data(
    html_path: Path,
    data_path: Path,
    enable_chunking: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Extract text content from Sphinx HTML file (standalone for multiprocessing)

    Args:
        html_path: Path to HTML file
        data_path: Base data path for relative path calculation
        enable_chunking: Whether to split content into chunks
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks

    Returns:
        List of dictionaries with extracted content (one per chunk, or single item if no chunking)
    """
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Extract title
        title = soup.find("title")
        title_text = title.get_text().strip() if title else html_path.stem

        # Remove scripts, styles and navigation
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Extract main content - Sphinx specific
        # Try different Sphinx content containers
        content_div = (
            soup.find("div", {"role": "main"})
            or soup.find("div", {"class": "document"})
            or soup.find("div", {"class": "body"})
            or soup.find("section")
        )

        if content_div:
            content = content_div.get_text(separator=" ", strip=True)
        else:
            content = soup.get_text(separator=" ", strip=True)

        # Extract code snippets
        code_snippets = []
        for code in soup.find_all(
            ["code", "pre", "div"], class_=re.compile(r"highlight|literal-block")
        ):
            snippet = code.get_text(strip=True)
            if snippet and len(snippet) > 10:
                code_snippets.append(snippet)

        # Determine category from path
        rel_path = html_path.relative_to(data_path)
        path_parts = rel_path.parts

        # Geant4 documentation categories
        category = "general"
        if "ForApplicationDevelopers" in str(rel_path):
            category = "application_developer"
        elif "PhysicsReferenceManual" in str(rel_path):
            category = "physics_reference"
        elif "PhysicsListGuide" in str(rel_path):
            category = "physics_list"
        elif "ForToolkitDeveloper" in str(rel_path):
            category = "toolkit_developer"
        elif "InstallationGuide" in str(rel_path):
            category = "installation"
        elif "FAQ" in str(rel_path):
            category = "faq"
        elif "IntroductionToGeant4" in str(rel_path):
            category = "introduction"

        # Determine content type
        content_type = "documentation"
        if "tutorial" in str(html_path).lower():
            content_type = "tutorial"
        elif "example" in str(html_path).lower():
            content_type = "example"
        elif "reference" in str(html_path).lower():
            content_type = "reference"

        # Build Geant4 documentation URL
        # Local path: etl/geant4/data/geant4/geant4-userdoc.web.cern.ch/UsersGuides/AllGuides/html/...
        # Target URL: https://geant4-userdoc.web.cern.ch/UsersGuides/AllGuides/html/...

        # Extract the path after the domain
        path_str = str(rel_path)
        if "geant4-userdoc.web.cern.ch" in path_str:
            # Remove the domain part and everything before it
            url_path = path_str.split("geant4-userdoc.web.cern.ch/", 1)[1]
        else:
            # Fallback: use relative path from html directory
            url_path = "UsersGuides/AllGuides/html/" + "/".join(path_parts)

        doc_url = f"https://geant4-userdoc.web.cern.ch/{url_path}"

        # Clean content
        content = clean_text(content)

        # Skip files with minimal content
        if len(content) < 200:
            return []

        # Create chunks if enabled
        if enable_chunking and len(content) > chunk_size:
            content_chunks = create_chunks(content, chunk_size, chunk_overlap)
        else:
            content_chunks = [content[:10000]]  # Single chunk, limit to 10k

        # Generate document ID base
        doc_id_base = hashlib.md5(str(html_path).encode()).hexdigest()

        # Create a document for each chunk
        documents = []
        for chunk_idx, chunk_content in enumerate(content_chunks):
            # Create unique ID for this chunk
            doc_id = f"{doc_id_base}_chunk_{chunk_idx}" if len(content_chunks) > 1 else doc_id_base

            documents.append(
                {
                    "doc_id": doc_id,
                    "title": title_text,
                    "content": chunk_content,
                    "code_snippets": (
                        code_snippets[:5] if chunk_idx == 0 else []
                    ),  # Only first chunk gets code
                    "category": category,
                    "content_type": content_type,
                    "file_path": str(rel_path),
                    "url": doc_url,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(content_chunks),
                    "is_chunked": len(content_chunks) > 1,
                }
            )

        return documents

    except Exception as e:
        logger.error(f"Error processing {html_path}: {e}")
        return []


class Geant4DocumentationIndexer:
    """Geant4 documentation indexer in Elasticsearch with embeddings"""

    def __init__(
        self,
        es_host: str = "http://localhost:9200",
        index_name: str = "geant4-documentation",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        data_path: str = "etl/geant4/data/geant4",
        enable_chunking: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the indexer

        Args:
            es_host: Elasticsearch URL
            index_name: Index name
            model_name: Embeddings model (recommended: all-MiniLM-L6-v2)
            data_path: Path to Geant4 data
            enable_chunking: Whether to split large documents into chunks
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlapping characters between chunks
        """
        self.es = Elasticsearch([es_host], request_timeout=60)
        self.index_name = index_name
        self.data_path = Path(data_path)
        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load embeddings model
        logger.info(f"Loading embeddings model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Verify connection
        try:
            info = self.es.info()
            logger.info(f"✓ Connected to Elasticsearch: {es_host}")
            logger.info(f"  Cluster: {info['cluster_name']}, Version: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise ConnectionError(f"Cannot connect to Elasticsearch at {es_host}: {e}")

    def create_index(self):
        """Create Elasticsearch index with mapping for vector search"""

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "code_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"],
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "code_analyzer",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "content": {"type": "text", "analyzer": "code_analyzer"},
                    "content_type": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "code_snippets": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "is_chunked": {"type": "boolean"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {"type": "object"},
                    "indexed_at": {"type": "date"},
                }
            },
        }

        # Delete index if exists
        if self.es.indices.exists(index=self.index_name):
            logger.warning(f"Deleting existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)

        # Create new index
        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"✓ Index created: {self.index_name}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts (more efficient)"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def process_documents(self, num_workers: int = None) -> Generator[Dict, None, None]:
        """
        Process all HTML documents with parallel HTML extraction and batched embedding generation

        Args:
            num_workers: Number of parallel workers for HTML extraction (defaults to CPU count)

        Yields:
            Documents ready for indexing
        """
        html_files = list(self.data_path.rglob("*.html")) + list(self.data_path.rglob("*.htm"))
        logger.info(f"Found {len(html_files)} HTML files")

        if num_workers is None:
            num_workers = cpu_count()

        logger.info(f"Extracting HTML content with {num_workers} parallel workers")

        # Extract HTML content in parallel with chunking support
        extract_func = partial(
            extract_html_data,
            data_path=self.data_path,
            enable_chunking=self.enable_chunking,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        with Pool(num_workers) as pool:
            html_results = list(
                tqdm(
                    pool.imap(extract_func, html_files, chunksize=10),
                    total=len(html_files),
                    desc="Extracting HTML",
                )
            )

        # Flatten list of lists (each file can produce multiple chunks)
        html_data_list = []
        for result in html_results:
            if result:  # result is a list of chunks
                html_data_list.extend(result)

        logger.info(
            f"Successfully extracted {len(html_data_list)} document chunks from {len(html_files)} files"
        )

        # Generate embeddings in batches (serial, but batched for efficiency)
        logger.info("Generating embeddings...")
        batch_size = 32

        for i in tqdm(range(0, len(html_data_list), batch_size), desc="Embedding generation"):
            batch = html_data_list[i : i + batch_size]

            # Prepare texts for embedding
            texts = [f"{doc['title']} {doc['content'][:2000]}" for doc in batch]

            # Generate embeddings for batch
            embeddings = self.generate_embeddings_batch(texts)

            # Yield complete documents with embeddings
            for doc_data, embedding in zip(batch, embeddings):
                document = {
                    "doc_id": doc_data["doc_id"],
                    "title": doc_data["title"],
                    "content": doc_data["content"],
                    "content_type": doc_data["content_type"],
                    "category": doc_data["category"],
                    "file_path": doc_data["file_path"],
                    "url": doc_data["url"],
                    "code_snippets": " ".join(doc_data["code_snippets"]),
                    "chunk_index": doc_data["chunk_index"],
                    "total_chunks": doc_data["total_chunks"],
                    "is_chunked": doc_data["is_chunked"],
                    "embedding": embedding,
                    "indexed_at": datetime.utcnow().isoformat(),
                }
                yield document

    def bulk_index_documents(self, batch_size: int = 100, num_workers: int = None):
        """
        Index documents in batches using bulk API with parallel processing

        Args:
            batch_size: Batch size for bulk indexing
            num_workers: Number of parallel workers for document processing
        """

        def generate_actions():
            for doc in self.process_documents(num_workers=num_workers):
                yield {"_index": self.index_name, "_id": doc["doc_id"], "_source": doc}

        # Batch indexing
        success_count = 0
        error_count = 0

        for success, info in helpers.parallel_bulk(
            self.es, generate_actions(), chunk_size=batch_size, thread_count=4, raise_on_error=False
        ):
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.error(f"Error indexing document: {info}")

        logger.info(f"✓ Indexing completed: {success_count} successful, {error_count} errors")

        # Refresh index
        self.es.indices.refresh(index=self.index_name)

    def search(self, query: str, k: int = 10, hybrid: bool = True) -> List[Dict]:
        """
        Hybrid search: vector + text

        Args:
            query: Search query
            k: Number of results
            hybrid: If True, combines vector and text search

        Returns:
            List of relevant documents
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        if hybrid:
            # Hybrid search (vector + text)
            search_query = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding},
                                    },
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "content^2", "code_snippets"],
                                    "type": "best_fields",
                                }
                            },
                        ]
                    }
                },
                "_source": ["title", "content", "url", "category", "content_type"],
            }
        else:
            # Vector search only
            search_query = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding},
                        },
                    }
                },
                "_source": ["title", "content", "url", "category", "content_type"],
            }

        response = self.es.search(index=self.index_name, body=search_query)

        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "score": hit["_score"],
                    "title": hit["_source"]["title"],
                    "content": hit["_source"]["content"][:500],  # First 500 chars
                    "url": hit["_source"]["url"],
                    "category": hit["_source"]["category"],
                    "type": hit["_source"]["content_type"],
                }
            )

        return results


def main():
    """Main function to execute indexing"""

    # Configuration
    # Use absolute path relative to script location
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "geant4"

    indexer = Geant4DocumentationIndexer(
        es_host="http://localhost:9200",
        index_name="geant4-documentation",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        data_path=str(data_path),
        enable_chunking=False,  # Set to True to enable chunking
    )

    # Create index
    logger.info("Creating Elasticsearch index...")
    indexer.create_index()

    # Index documents with parallel processing
    num_workers = cpu_count()
    logger.info(f"Starting document indexing with {num_workers} CPU cores...")
    indexer.bulk_index_documents(batch_size=100, num_workers=num_workers)

    # Statistics
    stats = indexer.es.count(index="geant4-documentation")
    logger.info(f"✓ Total documents indexed: {stats['count']}")

    # Search test
    logger.info("\n" + "=" * 60)
    logger.info("Search Test")
    logger.info("=" * 60)

    test_queries = [
        "How to create a detector geometry?",
        "Physics list examples",
        "Particle tracking tutorial",
    ]

    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = indexer.search(query, k=3)
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['title']} (score: {result['score']:.2f})")
            logger.info(f"     Category: {result['category']} | Type: {result['type']}")
            logger.info(f"     URL: {result['url']}")


if __name__ == "__main__":
    main()
