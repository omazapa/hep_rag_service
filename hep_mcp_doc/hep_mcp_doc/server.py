#!/usr/bin/env python3
"""
HEP MCP Documentation Server for RAG Testing

This MCP server provides tools to search and retrieve documentation
from HEP (High Energy Physics) projects like ROOT and Geant4.
"""

import asyncio
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import search functions from the ETL modules
try:
    from etl.root.index_root_docs import ROOTDocumentationIndexer
    from etl.geant4.index_geant4_docs import Geant4DocumentationIndexer
    HAS_INDEXERS = True
except ImportError:
    HAS_INDEXERS = False
    print("Warning: Could not import indexers. Make sure to install hep-rag-service package.")


# Initialize the MCP server
app = Server("hep-mcp-doc")


# Define available tools
@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for HEP documentation search."""
    return [
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


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "get_server_info":
        info = (
            "HEP MCP Documentation Server v0.1.0\n\n"
            "Available documentation sources:\n"
            "- ROOT: Data analysis framework (http://localhost:9200/root-documentation)\n"
            "- Geant4: Particle simulation toolkit (http://localhost:9200/geant4-documentation)\n\n"
            "Connection status:\n"
        )
        if HAS_INDEXERS:
            info += "✓ Indexers loaded successfully\n"
        else:
            info += "✗ Warning: Indexers not available. Install dependencies.\n"
        
        info += "\nUsage:\n"
        info += "- Use search_root_docs to search ROOT documentation\n"
        info += "- Use search_geant4_docs to search Geant4 documentation\n"
        
        return [TextContent(type="text", text=info)]
    
    elif name == "search_root_docs":
        if not HAS_INDEXERS:
            return [TextContent(
                type="text",
                text="Error: Indexers not available. Please install required dependencies."
            )]
        
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
            response += "\n" + "="*80 + "\n"
            response += "📚 REFERENCES:\n"
            response += "="*80 + "\n"
            for i, result in enumerate(results, 1):
                response += f"[{i}] {result['url']}\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching ROOT docs: {str(e)}")]
    
    elif name == "search_geant4_docs":
        if not HAS_INDEXERS:
            return [TextContent(
                type="text",
                text="Error: Indexers not available. Please install required dependencies."
            )]
        
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
            response += "\n" + "="*80 + "\n"
            response += "📚 REFERENCES:\n"
            response += "="*80 + "\n"
            for i, result in enumerate(results, 1):
                response += f"[{i}] {result['url']}\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching Geant4 docs: {str(e)}")]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


def main():
    """Entry point for the server."""
    async def _run():
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    
    asyncio.run(_run())


if __name__ == "__main__":
    main()
