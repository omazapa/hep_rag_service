#!/bin/bash

# Script to download Geant4 HTML documentation
# Usage: ./download_geant4_docs.sh

set -e  # Exit on error

# Configuration
BASE_URL="https://geant4-userdoc.web.cern.ch/UsersGuides/AllGuides/html"
DOWNLOAD_DIR="$(dirname "$0")/data"
TEMP_DIR="$DOWNLOAD_DIR/geant4"

echo "============================================================"
echo "Geant4 Documentation Download Script"
echo "============================================================"
echo ""

# Create directories
echo "📁 Creating directory structure..."
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TEMP_DIR"

# Download using wget with recursive option
echo "📥 Downloading Geant4 HTML documentation..."
echo "   URL: $BASE_URL"
echo "   Destination: $TEMP_DIR"
echo ""

# Wget options:
# -r: recursive
# -np: no parent (don't go up to parent directory)
# -A: accept only these file types (html and htm)
# -P: directory prefix
# --no-check-certificate: skip certificate check if needed
# -l: depth limit (10 levels should be enough)
# -e robots=off: ignore robots.txt

if wget -r -np \
     -P "$TEMP_DIR" \
     -A "*.html,*.htm" \
     -l 10 \
     -e robots=off \
     --no-check-certificate \
     --reject "index.html*\?*" \
     "$BASE_URL/"; then
    echo "✓ Download completed successfully!"
else
    echo "✗ Download failed!"
    exit 1
fi

# Move downloaded content to final destination
echo ""
echo "📦 Organizing downloaded files..."
SITE_DIR="$TEMP_DIR/geant4-userdoc.web.cern.ch/UsersGuides/AllGuides/html"

if [ -d "$SITE_DIR" ]; then
    mv "$SITE_DIR" "$DOWNLOAD_DIR/geant4"
    echo "✓ Files organized successfully!"
else
    echo "⚠️  Warning: Expected directory structure not found"
    echo "   Looking for alternative structure..."
    # Find the html directory wherever it is
    HTML_DIR=$(find "$TEMP_DIR" -name "html" -type d | head -1)
    if [ -n "$HTML_DIR" ]; then
        mv "$HTML_DIR" "$DOWNLOAD_DIR/geant4"
        echo "✓ Files moved successfully!"
    else
        echo "✗ Could not find HTML directory!"
        exit 1
    fi
fi

# Clean up temporary directory
echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Display statistics
echo ""
echo "============================================================"
echo "✓ Download completed!"
echo "============================================================"
echo ""
echo "📊 Statistics:"
echo "   Location: $DOWNLOAD_DIR/geant4"
echo "   Total HTML files: $(find "$DOWNLOAD_DIR/geant4" -type f -name "*.html" 2>/dev/null | wc -l)"
echo "   Total size: $(du -sh "$DOWNLOAD_DIR/geant4" 2>/dev/null | cut -f1)"
echo ""
echo "You can now run the indexing script:"
echo "   python etl/geant4/index_geant4_docs.py"
echo ""
