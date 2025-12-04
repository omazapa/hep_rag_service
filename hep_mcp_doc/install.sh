#!/bin/bash
# Quick installation script for hep_mcp_doc

echo "🚀 Installing hep_mcp_doc..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Install hep-rag-service first
echo "📥 Installing hep-rag-service..."
cd ..
pip install -e .

# Go back to MCP server folder
cd hep_mcp_doc

# Install the package
echo "📥 Installing hep-mcp-doc..."
pip install -e .

echo "✅ Installation completed!"
echo ""
echo "To use the MCP server, run:"
echo "  source venv/bin/activate"
echo "  hep-mcp-doc"
