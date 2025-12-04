# HEP MCP Documentation Server

MCP (Model Context Protocol) server for searching HEP (High Energy Physics) project documentation using RAG (Retrieval-Augmented Generation).

## Features

- 🔍 Search ROOT documentation using RAG
- 🔍 Search Geant4 documentation using RAG
- 🚀 Hybrid search (BM25 + vector)
- 📚 Elasticsearch integration for indexing
- 🛠️ Compatible with MCP protocol

## Installation

### Step 1: Install the main hep-rag-service package

First, you need to install the `hep-rag-service` package:

```bash
pip install -e .
```

### Step 2: Install the MCP server

```bash
cd hep_mcp_doc
pip install -e .
```

### With development dependencies

```bash
cd hep_mcp_doc
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.8
- Elasticsearch running at `http://localhost:9200`
- Documentation indices created (`root-documentation`, `geant4-documentation`)

## Usage

### Run the MCP server

```bash
hep-mcp-doc
```

Or directly with Python:

```bash
python -m hep_mcp_doc.server
```

### Available tools

The MCP server provides the following tools:

#### 1. `search_root_docs`

Searches ROOT documentation.

**Parameters:**
- `query` (string, required): Search query
- `k` (integer, optional): Number of results (default: 5)
- `hybrid` (boolean, optional): Use hybrid search (default: true)

**Example:**
```json
{
  "query": "How to create a TH1 histogram?",
  "k": 5,
  "hybrid": true
}
```

#### 2. `search_geant4_docs`

Searches Geant4 documentation.

**Parameters:**
- `query` (string, required): Search query
- `k` (integer, optional): Number of results (default: 5)
- `hybrid` (boolean, optional): Use hybrid search (default: true)

**Example:**
```json
{
  "query": "How to create a detector geometry?",
  "k": 5,
  "hybrid": true
}
```

#### 3. `get_server_info`

Gets information about the server and available documentation sources.

## MCP Client Configuration

To use this server with an MCP client, add the following configuration:

```json
{
  "mcpServers": {
    "hep-docs": {
      "command": "hep-mcp-doc"
    }
  }
}
```

## Project Structure

```
hep_mcp_doc/
├── hep_mcp_doc/
│   ├── __init__.py
│   └── server.py          # Main MCP server
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Development

### Run tests

```bash
pytest
```

### Format code

```bash
black hep_mcp_doc/
```

### Linter

```bash
ruff check hep_mcp_doc/
```

## Main Dependencies

- `mcp[cli]`: Model Context Protocol framework
- ETL modules from the main project for documentation search

## License

MIT

## Author

HEP RAG Service Team.Zapata@cern.ch
