# HEP RAG Service

A Retrieval-Augmented Generation (RAG) system for High Energy Physics (HEP) documentation, specifically designed to index and search ROOT and Geant4 framework documentation using Elasticsearch and semantic embeddings.

## 📚 Supported Frameworks

| Framework | Documents | Index Name | Documentation Type | Status |
|-----------|-----------|------------|-------------------|--------|
| **ROOT** | 25,804 docs (27,060 files) | `root-documentation` | Doxygen HTML | ✅ Active |
| **Geant4** | 333 docs (373 files) | `geant4-documentation` | Sphinx HTML | ✅ Active |

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This service provides semantic search capabilities over HEP framework documentation using:

- **Elasticsearch 8.17.4** for storage and hybrid search (vector + text)
- **sentence-transformers/all-MiniLM-L6-v2** for generating 384-dimensional embeddings
- **Parallel processing** with 20 CPU cores for fast indexing
- **Intelligent chunking** to split large documents into manageable pieces

### Framework Comparison

| Aspect | ROOT | Geant4 |
|--------|------|--------|
| **Format** | Doxygen HTML | Sphinx HTML |
| **Index** | `root-documentation` | `geant4-documentation` |
| **HTML Containers** | `div#doc-content` | `div[role='main']`, `section` |
| **Categories** | html, macros, pyzdoc, notebooks | app_dev, physics, installation, faq |
| **URL Base** | `root.cern/doc/master/` | `geant4-userdoc.web.cern.ch/` |
| **Documents** | 25,804 indexed (27,060 files) | 333 indexed (373 files) |

## ✨ Features

- 🚀 **Fast Indexing**: Processes ~27K documents in ~15 minutes with parallel HTML extraction
- 🔍 **Hybrid Search**: Combines semantic vector search with traditional text search
- 📚 **Intelligent Chunking**: Splits documents at sentence boundaries with configurable overlap
- 🌐 **Category-Aware URLs**: Generates correct documentation URLs based on content type
- ⚡ **GPU Acceleration**: Uses CUDA for faster embedding generation
- 🎯 **High Precision**: Returns relevant results with score-based ranking

## 🏗️ Architecture

```
┌─────────────────┐
│   ROOT Docs     │
│  (HTML Files)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   HTML Extraction (Parallel)        │
│   - 20 CPU workers                  │
│   - BeautifulSoup parsing           │
│   - Content cleaning                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Chunking (Sentence-aware)         │
│   - 1000 chars per chunk            │
│   - 200 chars overlap               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Embedding Generation (Batched)    │
│   - sentence-transformers           │
│   - 32 docs per batch               │
│   - GPU acceleration                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Elasticsearch Indexing            │
│   - Dense vector field (384 dims)   │
│   - Text fields for hybrid search   │
│   - Bulk API for performance        │
└─────────────────────────────────────┘
```

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **Docker**: For running Elasticsearch (or Kubernetes)
- **CUDA**: Optional, for GPU acceleration
- **System RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: ~5GB for documentation + ~2GB for Elasticsearch index

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/omazapa/hep_rag_service.git
cd hep_rag_service
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `elasticsearch>=8.0.0,<9.0.0` - Elasticsearch client
- `sentence-transformers>=2.2.0` - Embedding generation
- `torch>=2.0.0` - Deep learning framework
- `beautifulsoup4>=4.12.0` - HTML parsing
- `tqdm>=4.65.0` - Progress bars

### 3. Start Elasticsearch

**Option A: Docker (Recommended for development)**

```bash
cd servers
docker-compose up -d
```

**Option B: Kubernetes (Production)**

```bash
cd servers
kubectl apply -f k8s-elasticsearch.yaml
```

Verify Elasticsearch is running:

```bash
curl http://localhost:9200
```

## 🚀 Quick Start

### ROOT Framework

#### 1. Download ROOT Documentation

```bash
./etl/root/download_root_docs.sh
```

This will:
- Download `htmlmaster.tar.gz` from root.cern
- Extract to `etl/root/data/root/master/`
- Show statistics about downloaded files

#### 2. Index the Documentation

```bash
python etl/root/index_root_docs.py
```

Expected output:
- **Processing time**: ~15 minutes
- **Documents indexed**: ~25,804 documents from 27,060 files
- **Success rate**: 100%

#### 3. Search the Documentation

**Interactive Mode:**

```bash
python etl/root/search_root_docs.py
```

**Command Line:**

```bash
python etl/root/search_root_docs.py "How to create a TCanvas?"
```

**Test Suite:**

```bash
python etl/root/test_search.py
```

### Geant4 Framework

#### 1. Download Geant4 Documentation

**Sphinx Documentation (User Guides):**

```bash
./etl/geant4/download_geant4_docs.sh
```

This will:
- Download HTML documentation from geant4-userdoc.web.cern.ch
- Extract to `etl/geant4/data/geant4/`
- Process ~373 HTML files

**Doxygen Documentation (API Reference):**

```bash
python etl/geant4/download_geant4_doxygen.py
```

This will:
- Download API documentation from geant4.kek.jp/Reference/11.3.2/
- Extract to `etl/geant4/data/geant4_doxygen/`
- Process ~8,102 HTML files (240 MB)
- Download class, file, and namespace documentation

#### 2. Index the Documentation

```bash
python etl/geant4/index_geant4_docs.py
```

Expected output:
- **Processing time**: ~10 seconds (Sphinx) + ~5 minutes (Doxygen)
- **Documents indexed**: 333 documents from 373 files (Sphinx)
- **Success rate**: 100%

#### 3. Search the Documentation

**Interactive Mode:**

```bash
python etl/geant4/search_geant4_docs.py
```

**Command Line:**

```bash
python etl/geant4/search_geant4_docs.py "How to create a detector geometry?"
```

**Test Suite:**

```bash
python etl/geant4/test_search.py
```

## 📖 Usage

### Indexing Options

The indexer supports several configuration options:

```python
# In etl/root/index_root_docs.py

# Chunking configuration
ENABLE_CHUNKING = True      # Enable/disable document splitting
CHUNK_SIZE = 1000           # Characters per chunk
CHUNK_OVERLAP = 200         # Overlapping characters

# Processing configuration
NUM_WORKERS = 20            # Parallel HTML extraction workers
BATCH_SIZE = 32             # Documents per embedding batch
```

### Search Examples

**1. Basic Search**

```python
from etl.root.search_root_docs import search_documents

results = search_documents("TTree tutorial", top_k=5)
for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Score: {result['score']}")
```

**2. Advanced Search with Filters**

```python
results = search_documents(
    query="histogram examples",
    top_k=10,
    category_filter="html"  # Filter by category
)
```

**3. Hybrid Search (Vector + Text)**

The search automatically combines:
- **Vector similarity**: Semantic understanding using embeddings
- **BM25 text search**: Keyword matching for precision

## ⚙️ Configuration

### Elasticsearch Settings

**Docker Compose** (`servers/docker-compose.yml`):

```yaml
environment:
  - discovery.type=single-node
  - xpack.security.enabled=false
  - "ES_JAVA_OPTS=-Xms2g -Xmx4g"  # Memory allocation
```

**Kubernetes** (`servers/k8s-elasticsearch.yaml`):

```yaml
resources:
  requests:
    memory: "2Gi"
  limits:
    memory: "4Gi"
```

### Index Mapping

The Elasticsearch index uses:

```json
{
  "mappings": {
    "properties": {
      "embedding": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "cosine"
      },
      "content": { "type": "text" },
      "title": { "type": "text" },
      "category": { "type": "keyword" },
      "content_type": { "type": "keyword" }
    }
  }
}
```

## 📊 Performance

### Indexing Performance

| Metric | Value |
|--------|-------|
| Total Files | 27,060 |
| Total Chunks | 342,202 |
| Average Chunks/File | 12.6 |
| HTML Extraction Speed | ~264 files/sec |
| Embedding Generation | ~12.3 batches/sec |
| Total Indexing Time | ~15 minutes |
| Success Rate | 100% |

### Search Performance

| Query Type | Latency | Accuracy |
|------------|---------|----------|
| Simple queries | <100ms | High |
| Complex queries | <200ms | High |
| Bulk search (10 queries) | <500ms | High |

### Resource Usage

- **RAM**: ~4GB during indexing, ~2GB during search
- **GPU**: Optional, reduces indexing time by ~40%
- **Disk**: ~2GB for Elasticsearch index
- **CPU**: Scales linearly with worker count

## 📁 Project Structure

```
hep_rag_service/
├── etl/
│   └── root/
│       ├── data/
│       │   └── root/
│       │       └── master/          # ROOT documentation (27K+ files)
│       │           ├── html/        # HTML documentation
│       │           ├── macros/      # ROOT macros
│       │           ├── notebooks/   # Jupyter notebooks
│       │           └── pyzdoc/      # Python API docs
│       ├── download_root_docs.sh    # Download script
│       ├── index_root_docs.py       # Main indexing script
│       ├── search_root_docs.py      # Search interface
│       ├── test_search.py           # Search test suite
│       ├── test_url_generation.py   # URL validation
│       └── validate_urls.py         # URL verification
├── servers/
│   ├── docker-compose.yml       # Docker deployment
│   └── k8s-elasticsearch.yaml   # Kubernetes deployment
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔌 API Reference

### ROOTDocumentationIndexer

Main class for indexing ROOT documentation.

```python
from etl.root.index_root_docs import ROOTDocumentationIndexer

indexer = ROOTDocumentationIndexer(
    es_host="localhost",
    es_port=9200,
    index_name="root-documentation"
)

# Index all documents
indexer.index_all_documents(
    data_path="etl/root/data/root/master",
    enable_chunking=True,
    chunk_size=1000,
    chunk_overlap=200
)
```

**Methods:**

- `create_index()`: Creates Elasticsearch index with proper mapping
- `process_documents()`: Parallel HTML extraction and embedding generation
- `bulk_index_documents()`: Batch insert to Elasticsearch
- `index_all_documents()`: Full indexing pipeline

### Search Functions

```python
from etl.root.search_root_docs import search_documents

# Basic search
results = search_documents(
    query="TCanvas drawing",
    top_k=5,
    es_host="localhost",
    es_port=9200,
    index_name="root-documentation"
)

# Each result contains:
# - title: Document title
# - content: Text content (or chunk)
# - url: ROOT documentation URL
# - category: Content category (html, macros, notebooks, pyzdoc)
# - content_type: Document type
# - score: Relevance score
```

### URL Generation

The system generates category-aware URLs:

| Category | URL Pattern | Example |
|----------|-------------|---------|
| html | `https://root.cern/doc/master/{filename}` | `TCanvas_8h.html` |
| macros | `https://root.cern/doc/master/macros/{filename}` | `hist.html` |
| notebooks | `https://root.cern/doc/master/notebooks/{filename}` | `example.html` |
| pyzdoc | `https://root.cern/doc/master/pyzdoc/{filename}` | `_roofit.html` |

## 🐛 Troubleshooting

### Common Issues

**1. Elasticsearch Connection Failed**

```bash
# Check if Elasticsearch is running
curl http://localhost:9200

# Restart Elasticsearch
docker-compose restart

# Check logs
docker-compose logs elasticsearch
```

**2. Out of Memory During Indexing**

```python
# Reduce batch size in index_root_docs.py
BATCH_SIZE = 16  # Instead of 32

# Or reduce number of workers
NUM_WORKERS = 10  # Instead of 20
```

**3. CUDA Out of Memory**

```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

**4. Slow Indexing**

```bash
# Check CPU usage
htop

# Increase workers if CPU < 80%
NUM_WORKERS = 30  # Adjust based on your CPU

# Enable GPU acceleration
# Ensure PyTorch with CUDA is installed
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Validation Tools

**Test URL Generation:**

```bash
python etl/root/test_url_generation.py
```

**Test Search:**

```bash
python etl/root/test_search.py
```

**Check Index Statistics:**

```bash
curl http://localhost:9200/root-documentation/_count
curl http://localhost:9200/root-documentation/_stats
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Include type hints where applicable
- Write tests for new features
- Update documentation as needed

## 📝 License

This project is part of CERN's ROOT framework ecosystem. Please refer to ROOT's licensing terms.

## 🙏 Acknowledgments

- **ROOT Team** at CERN for the excellent documentation
- **sentence-transformers** team for the embedding models
- **Elasticsearch** team for the powerful search engine

## 📧 Contact

- **Repository**: https://github.com/omazapa/hep_rag_service
- **Issues**: https://github.com/omazapa/hep_rag_service/issues

## 🔗 Related Resources

- [ROOT Documentation](https://root.cern/doc/master/)
- [ROOT Forum](https://root-forum.cern.ch/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [sentence-transformers Documentation](https://www.sbert.net/)

---

**Built with ❤️ for the HEP community**

- Soporta ~28,000 documentos de ROOT

### `etl/search_root_docs.py`
Interfaz de búsqueda que implementa:
- Búsqueda híbrida (vectorial + texto)
- Modo interactivo
- Ejemplos predefinidos

## 🎯 Modelo de Embeddings

**Recomendado:** `sentence-transformers/all-MiniLM-L6-v2`

**Características:**
- 384 dimensiones
- 80MB de tamaño
- Excelente para documentación técnica
- Búsqueda semántica precisa
- Gratis y open source

**Alternativas:**
- `text-embedding-3-small` (OpenAI) - Mejor calidad, requiere API key
- `all-mpnet-base-v2` - Más preciso pero más lento
- `paraphrase-MiniLM-L6-v2` - Optimizado para paráfrasis

## 📊 Estructura de Datos

Los datos de ROOT están en `etl/root/data/root/master/`:
```
master/
├── html/        # Documentación HTML Doxygen (~20k archivos)
├── macros/      # Ejemplos de macros C++
├── notebooks/   # Jupyter notebooks
└── pyzdoc/      # Documentación Python
```

## 🔍 Ejemplo de Búsqueda

```python
from etl.root.index_root_docs import ROOTDocumentationIndexer

indexer = ROOTDocumentationIndexer()
results = indexer.search("How to create a histogram?", k=5)

for result in results:
    print(f"{result['title']}: {result['url']}")
```

## 🛠️ Configuración

### Variables de Elasticsearch

- **Host:** `http://localhost:30920` (NodePort)
- **Índice:** `root-documentation`
- **Timeout:** 60 segundos

### Parámetros de Indexación

- **Batch size:** 100 documentos
- **Threads:** 4 workers paralelos
- **Max content:** 10,000 caracteres por documento
- **Max snippets:** 5 por documento

## 📈 Rendimiento

- **Documentos:** ~28,000
- **Tiempo de indexación:** ~30-60 minutos (depende del hardware)
- **Búsqueda:** < 1 segundo
- **Memoria requerida:** ~4Gi para Elasticsearch

## 🔧 Troubleshooting

### Elasticsearch no inicia
```bash
kubectl logs -n elasticsearch deployment/elasticsearch
kubectl describe pod -n elasticsearch
```

### Error de memoria
Ajusta los recursos en `k8s-elasticsearch.yaml`:
```yaml
resources:
  requests:
    memory: "4Gi"
  limits:
    memory: "8Gi"
```

### Puerto no accesible
Verifica el servicio NodePort:
```bash
kubectl get svc -n elasticsearch
```

## 📝 Próximos Pasos

1. **Integrar LLM:** Conectar con GPT/Claude para respuestas generadas
2. **Fine-tuning:** Ajustar embeddings para vocabulario HEP
3. **UI Web:** Crear interfaz web para búsqueda
4. **API REST:** Exponer endpoints de búsqueda
5. **Métricas:** Agregar monitoreo y analytics

## 📄 Licencia

MIT License - CERN ROOT Documentation RAG System
