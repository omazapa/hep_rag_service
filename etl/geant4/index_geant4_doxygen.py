#!/usr/bin/env python3
"""
RAG System for Geant4 Doxygen Documentation - Elasticsearch Indexing
Processes Doxygen HTML documentation and indexes in Elasticsearch with vector embeddings
"""

import hashlib
import logging
import re
import sys
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

# Default Geant4 version
DEFAULT_VERSION = "11.3.2"


# Chunking utilities
def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace"""
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
            search_start = max(start, end - 100)
            last_period = text.rfind(".", search_start, end)
            last_newline = text.rfind("\n", search_start, end)
            last_question = text.rfind("?", search_start, end)
            last_exclamation = text.rfind("!", search_start, end)

            break_point = max(last_period, last_newline, last_question, last_exclamation)
            if break_point > start:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else end

    return chunks


# Standalone function for parallel HTML extraction
def extract_doxygen_html_data(
    html_path: Path,
    data_path: Path,
    base_url: str,
    enable_chunking: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Extract text content from Doxygen HTML file (standalone for multiprocessing)

    Args:
        html_path: Path to HTML file
        data_path: Base data path for relative path calculation
        base_url: Base URL for documentation (e.g., https://geant4.kek.jp/Reference/11.3.2/)
        enable_chunking: Whether to split content into chunks
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks

    Returns:
        List of dictionaries with extracted content
    """
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Extract title
        title = soup.find("title")
        title_text = title.get_text().strip() if title else html_path.stem

        # Remove unwanted elements first
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "iframe",
                "noscript",
                "form",
                "input",
                "button",
                "aside",
            ]
        ):
            tag.decompose()

        # Remove Doxygen navigation and UI elements (but keep memdoc/memproto for method extraction)
        for class_name in [
            "navpath",
            "navtab",
            "directory",
            "tabs",
            "tabs2",
            "tabs3",
            "search",
            "searchresults",
            "header",
            "headertitle",
            "dynheader",
            "dyncontent",
        ]:
            for tag in soup.find_all(class_=class_name):
                tag.decompose()

        # Remove SVG elements and their warnings
        for svg in soup.find_all(["svg", "map", "area"]):
            svg.decompose()

        # Remove "Loading..." and similar UI messages
        for element in soup.find_all(
            text=lambda text: text
            and (
                "Loading..." in text
                or "Searching..." in text
                or "No Matches" in text
                or "This browser is not able to show SVG" in text
                or "try Firefox, Chrome, Safari" in text
            )
        ):
            element.extract()

        # Extract structured Doxygen content - COMPLETE VERSION
        # Strategy: Extract all text preserving method/attribute information

        # Try main content containers
        content_div = (
            soup.find("div", {"class": "contents"})
            or soup.find("div", {"id": "doc-content"})
            or soup.find("body")
        )

        if not content_div:
            return []

        # Get ALL text content from the main div
        # This approach is simpler and captures everything
        content = content_div.get_text(separator=" ", strip=True)

        # Remove Geant4 license/copyright block (very different from ROOT)
        # Geant4 has a long license header (~24 lines) in source files
        import re

        # Remove the full Geant4 license block (from "License and Disclaimer" to end of license)
        # This matches the entire license header that appears in Geant4 source files
        geant4_license_pattern = r"(?:\/\/\s*\**\s*)?License and Disclaimer.*?acceptance of all terms of the Geant4 Software license\.\s*(?:\/\/\s*\**\s*)?"
        content = re.sub(geant4_license_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

        # Also remove shorter copyright mentions
        content = re.sub(
            r"The\s+Geant4\s+software\s+is\s+copyright\s+of\s+the\s+Copyright\s+Holders.*?\.",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r"copyright\s+of\s+the\s+Copyright\s+Holders\s+of\s+the\s+Geant4\s+Collaboration.*?\.",
            "",
            content,
            flags=re.IGNORECASE,
        )

        # Extract code snippets
        code_snippets = []
        for code in soup.find_all(["code", "pre", "div"], class_=re.compile(r"fragment|line")):
            snippet = code.get_text(strip=True)
            if snippet and len(snippet) > 10:
                code_snippets.append(snippet)

        # Determine category from filename pattern
        filename = html_path.name
        category = "general"
        content_type = "documentation"

        if filename.startswith("class"):
            category = "class"
            content_type = "class_reference"
        elif filename.startswith("struct"):
            category = "struct"
            content_type = "struct_reference"
        elif filename.startswith("namespace"):
            category = "namespace"
            content_type = "namespace_reference"
        elif filename.endswith("_8h.html") or filename.endswith("_8hh.html"):
            category = "header"
            content_type = "file_reference"
        elif filename.endswith("_8cc.html") or filename.endswith("_8cpp.html"):
            category = "source"
            content_type = "file_reference"
        elif filename in ["annotated.html", "classes.html", "hierarchy.html"]:
            category = "index"
            content_type = "index"
        elif filename in ["files.html", "globals.html", "functions.html"]:
            category = "index"
            content_type = "index"

        # Build Doxygen documentation URL
        # Local path: etl/geant4/data/geant4_doxygen/classG4Box.html
        # Target URL: https://geant4.kek.jp/Reference/11.3.2/classG4Box.html
        doc_url = f"{base_url}{filename}"

        # Clean content
        content = clean_text(content)

        # Skip files with minimal content
        if len(content) < 100:
            return []

        # Create chunks if enabled
        if enable_chunking and len(content) > chunk_size:
            content_chunks = create_chunks(content, chunk_size, chunk_overlap)
        else:
            # NO LIMIT - store complete content
            content_chunks = [content]

        # Generate document ID base
        doc_id_base = hashlib.md5(str(html_path).encode()).hexdigest()

        # Create a document for each chunk
        documents = []
        for chunk_idx, chunk_content in enumerate(content_chunks):
            doc_id = f"{doc_id_base}_chunk_{chunk_idx}" if len(content_chunks) > 1 else doc_id_base

            documents.append(
                {
                    "doc_id": doc_id,
                    "title": title_text,
                    "content": chunk_content,
                    "code_snippets": code_snippets[:5] if chunk_idx == 0 else [],
                    "category": category,
                    "content_type": content_type,
                    "file_path": filename,
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


class Geant4DoxygenIndexer:
    """Geant4 Doxygen documentation indexer in Elasticsearch with embeddings"""

    def __init__(
        self,
        es_host: str = "http://localhost:9200",
        index_name: str = "geant4-doxygen",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        data_path: str = None,
        geant4_version: str = DEFAULT_VERSION,
        enable_chunking: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the indexer

        Args:
            es_host: Elasticsearch URL
            index_name: Index name
            model_name: Embeddings model
            data_path: Path to Geant4 Doxygen data
            geant4_version: Geant4 version (for URL generation)
            enable_chunking: Whether to split large documents
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlapping characters between chunks
        """
        self.es = Elasticsearch([es_host], request_timeout=60)
        self.index_name = index_name
        self.geant4_version = geant4_version
        self.base_url = f"https://geant4.kek.jp/Reference/{geant4_version}/"

        # Set default data path if not provided
        if data_path is None:
            script_dir = Path(__file__).parent
            data_path = script_dir / "data" / "geant4_doxygen"

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
        """Generate embeddings for a batch of texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def process_documents(self, num_workers: int = None) -> Generator[Dict, None, None]:
        """
        Process all HTML documents with parallel extraction and batched embedding generation

        Args:
            num_workers: Number of parallel workers for HTML extraction

        Yields:
            Documents ready for indexing
        """
        html_files = list(self.data_path.glob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files")

        if num_workers is None:
            num_workers = cpu_count()

        logger.info(f"Extracting HTML content with {num_workers} parallel workers")

        # Extract HTML content in parallel
        extract_func = partial(
            extract_doxygen_html_data,
            data_path=self.data_path,
            base_url=self.base_url,
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

        # Flatten list of lists
        html_data_list = []
        for result in html_results:
            if result:
                html_data_list.extend(result)

        logger.info(
            f"Successfully extracted {len(html_data_list)} document chunks from {len(html_files)} files"
        )

        # Generate embeddings in batches
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
        Index documents in batches using bulk API

        Args:
            batch_size: Batch size for bulk indexing
            num_workers: Number of parallel workers
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
                    "content": hit["_source"]["content"][:500],
                    "url": hit["_source"]["url"],
                    "category": hit["_source"]["category"],
                    "type": hit["_source"]["content_type"],
                }
            )

        return results


def main():
    """Main function to execute indexing"""

    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print("\nUsage: python index_geant4_doxygen.py [VERSION]")
        print(f"\nDefault VERSION: {DEFAULT_VERSION}")
        print("\nThis will index Geant4 Doxygen documentation from data/geant4_doxygen/")
        print()
        sys.exit(0)

    version = DEFAULT_VERSION
    if len(sys.argv) > 1:
        version = sys.argv[1]
        logger.info(f"Using Geant4 version: {version}")

    # Configuration
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "geant4_doxygen"

    indexer = Geant4DoxygenIndexer(
        es_host="http://localhost:9200",
        index_name="geant4-doxygen",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        data_path=str(data_path),
        geant4_version=version,
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
    stats = indexer.es.count(index="geant4-doxygen")
    logger.info(f"✓ Total documents indexed: {stats['count']}")

    # Search test
    logger.info("\n" + "=" * 60)
    logger.info("Search Test")
    logger.info("=" * 60)

    test_queries = ["G4Box geometry", "G4VPhysicalVolume class", "How to define materials?"]

    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = indexer.search(query, k=3)
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['title']} (score: {result['score']:.2f})")
            logger.info(f"     Category: {result['category']} | Type: {result['type']}")
            logger.info(f"     URL: {result['url']}")


if __name__ == "__main__":
    main()
