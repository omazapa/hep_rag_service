#!/usr/bin/env python3
"""
Download Geant4 Doxygen documentation
Downloads all class and file documentation from https://geant4.kek.jp/Reference/
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Default version, can be overridden via command line
DEFAULT_VERSION = "11.3.2"

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "data" / "geant4_doxygen"

print("=" * 80)
print("Geant4 Doxygen Documentation Downloader")
print("=" * 80)


def download_file(url, output_path, session):
    """Download a single file"""
    try:
        if output_path.exists():
            return f"skip: {output_path.name}"

        response = session.get(url, timeout=30)
        response.raise_for_status()

        output_path.write_text(response.text, encoding="utf-8")
        return f"✓ {output_path.name}"
    except Exception as e:
        return f"✗ {output_path.name}: {str(e)[:50]}"


def get_linked_pages(html_content, pattern=None):
    """Extract linked HTML pages from content"""
    soup = BeautifulSoup(html_content, "html.parser")
    links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".html") and not href.startswith("http") and "#" not in href:
            if pattern is None or re.match(pattern, href):
                links.append(href)

    return list(set(links))


def main():
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print("\nUsage: python download_geant4_doxygen.py [VERSION]")
        print(f"\nDefault VERSION: {DEFAULT_VERSION}")
        print("\nExamples:")
        print(f"  python download_geant4_doxygen.py          # Uses {DEFAULT_VERSION}")
        print("  python download_geant4_doxygen.py 11.2.0   # Uses 11.2.0")
        print("  python download_geant4_doxygen.py 11.3.0   # Uses 11.3.0")
        print()
        sys.exit(0)

    version = DEFAULT_VERSION
    if len(sys.argv) > 1:
        version = sys.argv[1]
        print(f"Using Geant4 version: {version}")
    else:
        print(f"Using default Geant4 version: {version}")
        print(f"(To use a different version, run: python {os.path.basename(sys.argv[0])} <version>)")

    # Set up URLs and paths
    BASE_URL = f"https://geant4.kek.jp/Reference/{version}/"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nBase URL: {BASE_URL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})

    # Step 1: Download main index pages
    print("📥 Step 1: Downloading main index pages...")
    main_pages = [
        "index.html",
        "annotated.html",
        "classes.html",
        "hierarchy.html",
        "files.html",
        "namespaces.html",
        "namespacemembers.html",
        "functions.html",
        "globals.html",
    ]

    downloaded_files = set()

    for page in tqdm(main_pages, desc="Main pages"):
        url = urljoin(BASE_URL, page)
        output_path = OUTPUT_DIR / page
        download_file(url, output_path, session)
        downloaded_files.add(page)

    print(f"✓ Downloaded {len(main_pages)} main pages\n")

    # Step 2: Extract and download all class pages
    print("📥 Step 2: Downloading class documentation...")

    annotated_path = OUTPUT_DIR / "annotated.html"
    if annotated_path.exists():
        html_content = annotated_path.read_text(encoding="utf-8")
        class_pages = get_linked_pages(html_content, r"class.*\.html")

        print(f"   Found {len(class_pages)} class pages")

        # Download in parallel
        to_download = [p for p in class_pages if p not in downloaded_files]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for page in to_download:
                url = urljoin(BASE_URL, page)
                output_path = OUTPUT_DIR / page
                futures.append(executor.submit(download_file, url, output_path, session))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Classes"):
                future.result()
                # Optionally print results

        downloaded_files.update(to_download)

    print(f"✓ Downloaded class pages\n")

    # Step 3: Extract and download file documentation
    print("📥 Step 3: Downloading file documentation...")

    files_path = OUTPUT_DIR / "files.html"
    if files_path.exists():
        html_content = files_path.read_text(encoding="utf-8")
        file_pages = get_linked_pages(html_content, r".*\.(html)")
        file_pages = [p for p in file_pages if p not in downloaded_files]

        print(f"   Found {len(file_pages)} file pages")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for page in file_pages:
                url = urljoin(BASE_URL, page)
                output_path = OUTPUT_DIR / page
                futures.append(executor.submit(download_file, url, output_path, session))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Files"):
                future.result()

        downloaded_files.update(file_pages)

    print(f"✓ Downloaded file pages\n")

    # Step 4: Download namespace pages
    print("📥 Step 4: Downloading namespace documentation...")

    namespaces_path = OUTPUT_DIR / "namespaces.html"
    if namespaces_path.exists():
        html_content = namespaces_path.read_text(encoding="utf-8")
        namespace_pages = get_linked_pages(html_content, r"namespace.*\.html")
        namespace_pages = [p for p in namespace_pages if p not in downloaded_files]

        print(f"   Found {len(namespace_pages)} namespace pages")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for page in namespace_pages:
                url = urljoin(BASE_URL, page)
                output_path = OUTPUT_DIR / page
                futures.append(executor.submit(download_file, url, output_path, session))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Namespaces"):
                future.result()

        downloaded_files.update(namespace_pages)

    print(f"✓ Downloaded namespace pages\n")

    # Final statistics
    html_files = list(OUTPUT_DIR.glob("*.html"))
    total_size = sum(f.stat().st_size for f in html_files) / (1024 * 1024)  # MB

    print("=" * 80)
    print("✓ Download completed!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Location: {OUTPUT_DIR}")
    print(f"   Total HTML files: {len(html_files)}")
    print(f"   Total size: {total_size:.2f} MB")


if __name__ == "__main__":
    main()
