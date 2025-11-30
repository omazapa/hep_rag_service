#!/usr/bin/env python3
"""
Quick test script for ROOT documentation RAG system
"""

from index_root_docs import ROOTDocumentationIndexer
import sys


def test_search(query: str, k: int = 5):
    """
    Test search functionality
    
    Args:
        query: Search query
        k: Number of results to return
    """
    print("=" * 80)
    print(f"🔍 Searching ROOT Documentation")
    print("=" * 80)
    
    # Initialize indexer
    indexer = ROOTDocumentationIndexer(
        es_host="http://localhost:9200",
        index_name="root-documentation"
    )
    
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
        content = result['content'][:300].replace('\n', ' ')
        print(f"       {content}...")
        print()
    
    print("=" * 80)


def main():
    """Main function"""
    
    # Test queries
    test_queries = [
        "How to create and use TCanvas?",
        "TCanvas drawing options",
        "TCanvas divide pads",
        "TCanvas save to PDF",
        "TCanvas SetLogx SetLogy"
    ]
    
    if len(sys.argv) > 1:
        # Use query from command line
        query = " ".join(sys.argv[1:])
        test_search(query)
    else:
        # Run all test queries
        print("\n" + "=" * 80)
        print("🧪 Running TCanvas Test Queries")
        print("=" * 80 + "\n")
        
        for query in test_queries:
            test_search(query, k=3)
            print("\n")


if __name__ == "__main__":
    main()
