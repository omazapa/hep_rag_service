# HEP MCP Documentation Server

MCP (Model Context Protocol) server for searching HEP (High Energy Physics) project documentation using RAG (Retrieval-Augmented Generation).

## Features

- 🔍 Search ROOT documentation using RAG
- 🔍 Search Geant4 documentation using RAG
- 🤖 LangChain-powered agents with different personas (user/developer/teacher)
- 🚀 Hybrid search (BM25 + vector)
- 📚 Elasticsearch integration for indexing
- 🛠️ Compatible with MCP protocol
- 💬 Two server modes: basic retrieval or LLM-enhanced responses

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

**Basic server (retrieval only):**
```bash
hep-mcp-doc
```

**Server with LangChain agents (retrieval + LLM generation):**
```bash
hep-mcp-doc-agents
```

Or directly with Python:

```bash
# Basic server
python -m hep_mcp_doc.server

# Server with agents
python -m hep_mcp_doc.server_with_agents
```

### Available tools

**Basic server** provides:
- `search_root_docs`: Direct retrieval search
- `search_geant4_docs`: Direct retrieval search
- `get_server_info`: Server information

**Server with agents** provides all basic tools plus:
- `ask_root_user`: LangChain agent for practical ROOT help
- `ask_root_developer`: LangChain agent for technical/architectural questions
- `ask_root_teacher`: LangChain agent for educational explanations

#### 1. `search_root_docs`

Search ROOT documentation using RAG (Retrieval Augmented Generation).

**Parameters:**
- `query` (required): Natural language search query
- `k` (optional): Number of results to return (default: 5)
- `hybrid` (optional): Use hybrid search combining BM25 + vector search (default: true)

**Example:**
```json
{
  "query": "How do I create a histogram in ROOT?",
  "k": 5,
  "hybrid": true
}
```

#### 2. `search_geant4_docs`

Search Geant4 documentation using RAG.

**Parameters:**
- `query` (required): Natural language search query
- `k` (optional): Number of results to return (default: 5)
- `hybrid` (optional): Use hybrid search (default: true)

**Example:**
```json
{
  "query": "Explain particle transportation in Geant4",
  "k": 3
}
```

#### 3. `get_server_info`

Get information about the HEP MCP server and available documentation sources.

**Parameters:** None

#### 4. `ask_root_user` (agents only)

Get practical help with ROOT tasks from a friendly user assistant. Provides code examples and step-by-step guidance.

**Parameters:**
- `query` (required): Your ROOT question or problem
- `k` (optional): Number of documentation chunks to retrieve (default: 5)

**Example:**
```json
{
  "query": "I need to fit a Gaussian to my data, how do I do it?",
  "k": 5
}
```

#### 5. `ask_root_developer` (agents only)

Get technical and architectural guidance from an expert ROOT developer. Best for performance, design patterns, and advanced topics.

**Parameters:**
- `query` (required): Your technical question
- `k` (optional): Number of documentation chunks to retrieve (default: 7)

**Example:**
```json
{
  "query": "What's the best way to optimize ROOT I/O for large datasets?",
  "k": 7
}
```

#### 6. `ask_root_teacher` (agents only)

Get educational explanations from a patient tutor. Best for learning ROOT concepts from scratch.

**Parameters:**
- `query` (required): What you want to learn
- `k` (optional): Number of documentation chunks to retrieve (default: 5)

**Example:**
```json
{
  "query": "Explain what a TTree is and when I should use it",
  "k": 5
}
```

## Agent vs Basic Server

### Basic Server
- Direct retrieval from documentation
- Returns raw documentation chunks
- Fast and lightweight
- No LLM required
- Use when: You want exact documentation snippets

### Server with Agents
- LLM-powered responses using Ollama (llama3)
- Personalized by agent type (user/developer/teacher)
- Generates natural language answers
- Includes code examples and explanations
- Requires: Ollama running locally with llama3 model
- Use when: You want conversational help and generated content

## Requirements for Agent Server

To use the agent-enabled server (`hep-mcp-doc-agents`):

1. **Ollama with llama3 model:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3 model
ollama pull llama3

# Verify it's running
ollama list
```

2. **Elasticsearch with indexed documentation:**
```bash
# Make sure Elasticsearch is running on localhost:9200
# Index should contain ROOT documentation
```

The basic server works without these requirements.

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
