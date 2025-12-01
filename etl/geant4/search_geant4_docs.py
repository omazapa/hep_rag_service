#!/usr/bin/env python3
"""
Search script for Geant4 documentation
"""

import sys

from index_geant4_docs import Geant4DocumentationIndexer


def search_geant4(query: str, k: int = 5):
    """
    Search Geant4 documentation

    Args:
        query: Search query
        k: Number of results to return
    """
    print("=" * 80)
    print(f"🔍 Searching Geant4 Documentation")
    print("=" * 80)

    # Initialize indexer
    indexer = Geant4DocumentationIndexer(es_host="http://localhost:9200", index_name="geant4-documentation")

    print(f"\nQuery: {query}")
    print("-" * 80)

    # Search
    results = indexer.search(query, k=k, hybrid=True)

    if not results:
        print("❌ No results found")
        return

    print(f"\n✓ Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['title']}")
        print(f"    🎯 Score: {result['score']:.3f}")
        print(f"    📁 Category: {result['category']} | Type: {result['type']}")
        print(f"    🔗 URL: {result['url']}")
        print(f"    📄 Content preview:")

        # Show first 300 characters of content
        content = result["content"][:300].replace("\n", " ")
        print(f"       {content}...")
        print()

    print("=" * 80)


def interactive_search():
    """Interactive search mode"""

    indexer = Geant4DocumentationIndexer(es_host="http://localhost:9200", index_name="geant4-documentation")

    print("=" * 80)
    print("🔍 Geant4 Documentation - Interactive Search")
    print("=" * 80)
    print("Type 'exit' to quit\n")

    while True:
        try:
            query = input("\n🔍 Your question: ").strip()

            if query.lower() in ["exit", "quit", "salir"]:
                print("Goodbye!")
                break

            if not query:
                continue

            results = indexer.search(query, k=5)

            print(f"\n{'─'*80}")
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result['title']}")
                print(f"    🎯 Score: {result['score']:.3f}")
                print(f"    📁 {result['category']} | {result['type']}")
                print(f"    🔗 {result['url']}")
                print(f"    📄 {result['content'][:300]}...")
            print(f"{'─'*80}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use query from command line
        query = " ".join(sys.argv[1:])
        search_geant4(query)
    else:
        # Interactive mode
        interactive_search()
