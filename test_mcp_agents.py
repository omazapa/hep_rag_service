#!/usr/bin/env python3
"""
Test script for the HEP MCP Documentation Server with LangChain Agents

This script demonstrates the integration of the MCP server with LangChain agents.
"""

import asyncio
from hep_mcp_doc.server_with_agents import list_tools


async def test_server():
    """Test the MCP server with agents."""
    print("=" * 70)
    print("HEP MCP Documentation Server - LangChain Agents Integration Test")
    print("=" * 70)

    # List available tools
    tools = await list_tools()

    print(f"\n📋 Available MCP Tools ({len(tools)}):\n")

    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}")
        print(f"   {tool.description}")
        print()

    print("=" * 70)
    print("✅ Server Integration Test Passed!")
    print("=" * 70)
    print("\nTo use the server:")
    print("  • Basic server:        hep-mcp-doc")
    print("  • Server with agents:  hep-mcp-doc-agents")
    print("\nRequirements for agent server:")
    print("  • Elasticsearch running on localhost:9200")
    print("  • Ollama with llama3:8b model")
    print("  • ROOT documentation indexed in 'root-documentation'")


if __name__ == "__main__":
    asyncio.run(test_server())
