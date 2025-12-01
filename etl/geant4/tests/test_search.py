#!/usr/bin/env python3
"""
Test script for Geant4 documentation search
"""

from index_geant4_docs import Geant4DocumentationIndexer


def test_geant4_search():
    """Test search functionality with sample queries"""

    indexer = Geant4DocumentationIndexer(
        es_host="http://localhost:9200", index_name="geant4-documentation"
    )

    print("=" * 80)
    print("🧪 Geant4 Documentation Search Tests")
    print("=" * 80)

    # Test queries for different categories
    test_queries = [
        # Detector geometry
        "How to create detector geometry?",
        "G4Box solid volume",
        "logical and physical volumes",
        # Physics
        "electromagnetic physics processes",
        "hadronic interactions",
        "physics list construction",
        # Particle tracking
        "particle tracking in Geant4",
        "step and track information",
        # Installation
        "how to install Geant4",
        "compilation requirements",
        # Examples
        "basic example B1",
        "detector construction example",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")

        results = indexer.search(query, k=3, hybrid=True)

        if not results:
            print("❌ No results found")
            continue

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['title']}")
            print(f"    🎯 Score: {result['score']:.3f}")
            print(f"    📁 Category: {result['category']}")
            print(f"    🔗 URL: {result['url']}")
            print(f"    📄 Preview: {result['content'][:200]}...")

    print(f"\n{'='*80}")
    print("✓ Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_geant4_search()
