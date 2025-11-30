#!/usr/bin/env python3
"""
Example script for searching ROOT documentation
"""

from index_root_docs import ROOTDocumentationIndexer
import json


def search_examples():
    """Search examples in ROOT documentation"""
    
    # Initialize indexer (for search only)
    indexer = ROOTDocumentationIndexer(
        es_host="http://localhost:9200",  # Docker: localhost:9200, K8s: localhost:30920
        index_name="root-documentation"
    )
    
    print("=" * 80)
    print("🔍 RAG SYSTEM - ROOT Documentation Search")
    print("=" * 80)
    
    # Search examples
    queries = [
        "How to create and fill a TH1 histogram?",
        "TTree branch tutorial",
        "RDataFrame filter and define columns",
        "How to fit a Gaussian distribution?",
        "TCanvas drawing options",
        "PyROOT pythonizations",
        "RooFit workspace tutorial",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = indexer.search(query, k=5, hybrid=True)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['title']}")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Category: {result['category']} | Type: {result['type']}")
            print(f"    URL: {result['url']}")
            print(f"    Content: {result['content'][:200]}...")
    
    # Interactive search
    print("\n" + "=" * 80)
    print("Interactive Mode")
    print("=" * 80)
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'salir']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            results = indexer.search(query, k=3)
            
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
    search_examples()
