#!/bin/bash

# Script to download and extract ROOT HTML documentation
# Usage: ./download_root_docs.sh

set -e  # Exit on error

# Configuration
URL="https://root.cern/download/htmlmaster.tar.gz"
DOWNLOAD_DIR="$(dirname "$0")/data/root"
TEMP_FILE="/tmp/htmlmaster.tar.gz"

echo "============================================================"
echo "ROOT Documentation Download Script"
echo "============================================================"
echo ""

# Create data directory if it doesn't exist
echo "📁 Creating directory structure..."
mkdir -p "$DOWNLOAD_DIR"

# Download the file
echo "📥 Downloading ROOT HTML documentation..."
echo "   URL: $URL"
echo "   Destination: $TEMP_FILE"
echo ""

if curl -L --progress-bar -o "$TEMP_FILE" "$URL"; then
    echo "✓ Download completed successfully!"
else
    echo "✗ Download failed!"
    exit 1
fi

# Extract the tarball
echo ""
echo "📦 Extracting tarball..."
echo "   From: $TEMP_FILE"
echo "   To: $DOWNLOAD_DIR"
echo ""

if tar -xzf "$TEMP_FILE" -C "$DOWNLOAD_DIR"; then
    echo "✓ Extraction completed successfully!"
else
    echo "✗ Extraction failed!"
    rm -f "$TEMP_FILE"
    exit 1
fi

# Clean up temporary file
echo ""
echo "🧹 Cleaning up temporary files..."
rm -f "$TEMP_FILE"

# Display statistics
echo ""
echo "============================================================"
echo "✓ Download and extraction completed!"
echo "============================================================"
echo ""
echo "📊 Statistics:"
echo "   Location: $DOWNLOAD_DIR"
echo "   Total files: $(find "$DOWNLOAD_DIR" -type f -name "*.html" | wc -l)"
echo "   Total size: $(du -sh "$DOWNLOAD_DIR" | cut -f1)"
echo ""
echo "You can now run the indexing script:"
echo "   python etl/index_root_docs.py"
echo ""
