#!/usr/bin/env python3
"""
HEP MCP Documentation Server with LangChain Agents

This MCP server provides tools to search and interact with HEP documentation
using both direct retrieval and LangChain-powered agents with different personas.
"""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import indexers
try:
    from etl.root.index_root_docs import ROOTDocumentationIndexer
    from etl.geant4.index_geant4_docs import Geant4DocumentationIndexer

    HAS_INDEXERS = True
except ImportError:
    HAS_INDEXERS = False
    print("Warning: Could not import indexers. Make sure to install hep-rag-service package.")

# Import LangChain agents
try:
    from agents import LangChainHEPIndexer, ROOTUserAgent, ROOTDeveloperAgent, ROOTTeachingAgent

    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    print("Warning: Could not import LangChain agents. Install langchain dependencies.")


# Initialize the MCP server
app = Server("hep-mcp-doc")

# Global agents (lazy initialization)
_user_agent = None
_developer_agent = None
_teaching_agent = None


def get_user_agent():
    """Lazy initialization of ROOT User Agent."""
    global _user_agent
    if _user_agent is None and HAS_AGENTS:
        indexer = LangChainHEPIndexer(es_host="http://localhost:9200", index_name="root-documentation")
        _user_agent = ROOTUserAgent(indexer)
    return _user_agent


def get_developer_agent():
    """Lazy initialization of ROOT Developer Agent."""
    global _developer_agent
    if _developer_agent is None and HAS_AGENTS:
        indexer = LangChainHEPIndexer(es_host="http://localhost:9200", index_name="root-documentation")
        _developer_agent = ROOTDeveloperAgent(indexer)
    return _developer_agent


def get_teaching_agent():
    """Lazy initialization of ROOT Teaching Agent."""
    global _teaching_agent
    if _teaching_agent is None and HAS_AGENTS:
        indexer = LangChainHEPIndexer(es_host="http://localhost:9200", index_name="root-documentation")
        _teaching_agent = ROOTTeachingAgent(indexer)
    return _teaching_agent


# Define available tools
@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for HEP documentation search."""
    tools = [
        Tool(
            name="search_root_docs",
            description=(
                "Search ROOT documentation using RAG. "
                "ROOT is a data analysis framework used in high energy physics. "
                "This tool searches through ROOT user guides, tutorials, and API documentation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (natural language question or keywords)",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                    "hybrid": {
                        "type": "boolean",
                        "description": "Use hybrid search (BM25 + vector search, default: true)",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_geant4_docs",
            description=(
                "Search Geant4 documentation using RAG. "
                "Geant4 is a toolkit for simulating the passage of particles through matter. "
                "This tool searches through Geant4 user guides, installation guides, and physics reference manuals."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (natural language question or keywords)",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                    "hybrid": {
                        "type": "boolean",
                        "description": "Use hybrid search (BM25 + vector search, default: true)",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_server_info",
            description="Get information about the HEP MCP server and available documentation sources.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]

    # Add LangChain agent tools if available
    if HAS_AGENTS:
        tools.extend(
            [
                Tool(
                    name="ask_root_user",
                    description=(
                        "Ask ROOT questions as a user seeking practical help. "
                        "This agent provides friendly, clear guidance with working code examples "
                        "for common ROOT tasks like histograms, trees, graphs, and fitting. "
                        "Best for users who want step-by-step instructions."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Your question about using ROOT",
                            },
                        },
                        "required": ["question"],
                    },
                ),
                Tool(
                    name="ask_root_developer",
                    description=(
                        "Ask technical ROOT questions as a developer. "
                        "This expert agent provides in-depth technical explanations about ROOT's "
                        "architecture, internals, performance optimization, and advanced features. "
                        "Best for developers working on ROOT-based applications or exploring internals."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Your technical question about ROOT",
                            },
                        },
                        "required": ["question"],
                    },
                ),
                Tool(
                    name="ask_root_teacher",
                    description=(
                        "Learn ROOT concepts from a patient teacher. "
                        "This tutor agent explains ROOT from first principles with clear analogies, "
                        "simple examples, and progressive learning paths. "
                        "Best for newcomers to ROOT or HEP data analysis."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Your learning question about ROOT",
                            },
                        },
                        "required": ["question"],
                    },
                ),
            ]
        )

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""

    if name == "get_server_info":
        info = (
            "HEP MCP Documentation Server v1.1.0\n\n"
            "Available documentation sources:\n"
            "- ROOT: Data analysis framework (http://localhost:9200/root-documentation)\n"
            "- Geant4: Particle simulation toolkit (http://localhost:9200/geant4-documentation)\n\n"
            "Connection status:\n"
        )
        if HAS_INDEXERS:
            info += "✓ Indexers loaded successfully\n"
        else:
            info += "✗ Warning: Indexers not available. Install dependencies.\n"

        if HAS_AGENTS:
            info += "✓ LangChain agents available\n"
        else:
            info += "✗ Warning: LangChain agents not available. Install langchain dependencies.\n"

        info += "\nAvailable tools:\n"
        info += "- search_root_docs: Direct retrieval search in ROOT docs\n"
        info += "- search_geant4_docs: Direct retrieval search in Geant4 docs\n"

        if HAS_AGENTS:
            info += "\nLangChain Agents (RAG with LLM generation):\n"
            info += "- ask_root_user: Practical help for ROOT users\n"
            info += "- ask_root_developer: Technical guidance for developers\n"
            info += "- ask_root_teacher: Educational explanations for learners\n"

        return [TextContent(type="text", text=info)]

    elif name == "search_root_docs":
        if not HAS_INDEXERS:
            return [
                TextContent(type="text", text="Error: Indexers not available. Please install required dependencies.")
            ]

        query = arguments.get("query")
        k = arguments.get("k", 5)
        hybrid = arguments.get("hybrid", True)

        try:
            # Initialize ROOT indexer
            indexer = ROOTDocumentationIndexer(
                es_host="http://localhost:9200",
                index_name="root-documentation",
            )

            # Perform search
            results = indexer.search(query, k=k, hybrid=hybrid)

            if not results:
                return [TextContent(type="text", text=f"No results found for query: {query}")]

            # Format results
            response = f"Found {len(results)} results for: {query}\n\n"

            for i, result in enumerate(results, 1):
                response += f"[{i}] {result['title']}\n"
                response += f"    Score: {result['score']:.3f}\n"
                response += f"    Category: {result['category']} | Type: {result['type']}\n"
                response += f"    📖 {result['url']}\n"
                response += f"    Content:\n"
                response += f"    {result['content'][:400]}...\n\n"

            # Add references section
            response += "\n" + "=" * 80 + "\n"
            response += "📚 REFERENCES:\n"
            response += "=" * 80 + "\n"
            for i, result in enumerate(results, 1):
                response += f"[{i}] {result['url']}\n"

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error searching ROOT docs: {str(e)}")]

    elif name == "search_geant4_docs":
        if not HAS_INDEXERS:
            return [
                TextContent(type="text", text="Error: Indexers not available. Please install required dependencies.")
            ]

        query = arguments.get("query")
        k = arguments.get("k", 5)
        hybrid = arguments.get("hybrid", True)

        try:
            # Initialize Geant4 indexer
            indexer = Geant4DocumentationIndexer(
                es_host="http://localhost:9200",
                index_name="geant4-documentation",
            )

            # Perform search
            results = indexer.search(query, k=k, hybrid=hybrid)

            if not results:
                return [TextContent(type="text", text=f"No results found for query: {query}")]

            # Format results
            response = f"Found {len(results)} results for: {query}\n\n"

            for i, result in enumerate(results, 1):
                response += f"[{i}] {result['title']}\n"
                response += f"    Score: {result['score']:.3f}\n"
                response += f"    Category: {result['category']} | Type: {result['type']}\n"
                response += f"    📖 {result['url']}\n"
                response += f"    Content:\n"
                response += f"    {result['content'][:400]}...\n\n"

            # Add references section
            response += "\n" + "=" * 80 + "\n"
            response += "📚 REFERENCES:\n"
            response += "=" * 80 + "\n"
            for i, result in enumerate(results, 1):
                response += f"[{i}] {result['url']}\n"

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error searching Geant4 docs: {str(e)}")]

    # LangChain Agent tools
    elif name == "ask_root_user":
        if not HAS_AGENTS:
            return [
                TextContent(
                    type="text", text="Error: LangChain agents not available. Please install langchain dependencies."
                )
            ]

        question = arguments.get("question")

        try:
            agent = get_user_agent()
            result = agent.ask(question)

            # Format response with answer and sources
            response = f"🤝 ROOT User Assistant\n\n"
            response += f"{result['answer']}\n\n"
            response += "=" * 80 + "\n"
            response += "📚 SOURCES:\n"
            response += "=" * 80 + "\n"
            for i, source in enumerate(result["sources"], 1):
                response += f"[{i}] {source['title']} - {source['url']}\n"

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error with ROOT user agent: {str(e)}")]

    elif name == "ask_root_developer":
        if not HAS_AGENTS:
            return [
                TextContent(
                    type="text", text="Error: LangChain agents not available. Please install langchain dependencies."
                )
            ]

        question = arguments.get("question")

        try:
            agent = get_developer_agent()
            result = agent.ask(question)

            # Format response with answer and sources
            response = f"💻 ROOT Developer Expert\n\n"
            response += f"{result['answer']}\n\n"
            response += "=" * 80 + "\n"
            response += "📚 SOURCES:\n"
            response += "=" * 80 + "\n"
            for i, source in enumerate(result["sources"], 1):
                response += f"[{i}] {source['title']} - {source['url']}\n"

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error with ROOT developer agent: {str(e)}")]

    elif name == "ask_root_teacher":
        if not HAS_AGENTS:
            return [
                TextContent(
                    type="text", text="Error: LangChain agents not available. Please install langchain dependencies."
                )
            ]

        question = arguments.get("question")

        try:
            agent = get_teaching_agent()
            result = agent.ask(question)

            # Format response with answer and sources
            response = f"📚 ROOT Teacher\n\n"
            response += f"{result['answer']}\n\n"
            response += "=" * 80 + "\n"
            response += "📖 LEARNING RESOURCES:\n"
            response += "=" * 80 + "\n"
            for i, source in enumerate(result["sources"], 1):
                response += f"[{i}] {source['title']} - {source['url']}\n"

            return [TextContent(type="text", text=response)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error with ROOT teaching agent: {str(e)}")]

    else:
        raise ValueError(f"Unknown tool: {name}")


def main():
    """Entry point for the server."""

    async def _run():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
