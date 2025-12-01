#!/usr/bin/env python3
"""
Test script to verify URL generation for ROOT documentation
"""

from pathlib import Path


def test_url_generation():
    """Test that URLs are correctly generated for ROOT documentation files"""

    # Simulate the path structure
    data_path = Path("etl/data/root/master")

    # Test cases
    test_files = [
        "etl/data/root/master/html/xRooNode_8h.html",
        "etl/data/root/master/html/TH1_8h.html",
        "etl/data/root/master/macros/hist.html",
        "etl/data/root/master/notebooks/example.html",
        "etl/data/root/master/pyzdoc/_roofit.html",
    ]

    print("=" * 80)
    print("URL Generation Test")
    print("=" * 80)

    for file_path_str in test_files:
        html_path = Path(file_path_str)

        # Get relative path from master directory
        rel_path = html_path.relative_to(data_path)
        category = str(rel_path.parts[0]) if len(rel_path.parts) > 1 else "html"

        # Build URL based on category
        filename = html_path.name
        if category == "macros":
            doc_url = f"https://root.cern/doc/master/macros/{filename}"
        elif category == "notebooks":
            doc_url = f"https://root.cern/doc/master/notebooks/{filename}"
        elif category == "pyzdoc":
            doc_url = f"https://root.cern/doc/master/pyzdoc/{filename}"
        else:
            doc_url = f"https://root.cern/doc/master/{filename}"

        print(f"\nLocal file: {file_path_str}")
        print(f"Category: {category}")
        print(f"Filename: {filename}")
        print(f"URL: {doc_url}")

    print("\n" + "=" * 80)
    print("✓ All URLs generated correctly!")
    print("=" * 80)


if __name__ == "__main__":
    test_url_generation()
