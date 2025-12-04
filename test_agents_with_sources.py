#!/usr/bin/env python3
"""
Quick test to verify agents return sources correctly
"""

from agents import LangChainHEPIndexer, ROOTUserAgent

def main():
    print("Testing agents with sources...")
    
    # Initialize indexer
    indexer = LangChainHEPIndexer(
        es_host="http://localhost:9200",
        index_name="root-documentation"
    )
    
    # Create agent
    agent = ROOTUserAgent(indexer.as_retriever(search_kwargs={"k": 3}))
    
    # Ask a question
    print("\nAsking: 'How do I create a histogram in ROOT?'\n")
    result = agent.ask("How do I create a histogram in ROOT?")
    
    # Print result structure
    print("=" * 80)
    print("RESULT STRUCTURE:")
    print("=" * 80)
    print(f"Keys in result: {list(result.keys())}")
    print(f"Number of sources: {len(result['sources'])}")
    
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result['answer'])
    
    print("\n" + "=" * 80)
    print("SOURCES:")
    print("=" * 80)
    for i, source in enumerate(result['sources'], 1):
        print(f"\n[{i}] {source['title']}")
        print(f"    🔗 {source['url']}")
        print(f"    📄 {source['content']}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()
