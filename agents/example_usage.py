#!/usr/bin/env python3
"""
Example usage of ROOT Agents

This script demonstrates how to use the different ROOT agent personas
to interact with ROOT documentation.
"""

from agents import LangChainHEPIndexer, ROOTUserAgent, ROOTDeveloperAgent, ROOTTeachingAgent


def main():
    """Example usage of ROOT agents."""

    # Initialize the indexer
    print("Initializing LangChain indexer...")
    indexer = LangChainHEPIndexer(es_host="http://localhost:9200", index_name="root-documentation")

    # Example questions
    user_question = "How do I create and fill a TH1F histogram in ROOT?"
    developer_question = (
        "What are the performance implications of using TTree vs RDataFrame for processing large datasets?"
    )
    teaching_question = "What is a histogram in ROOT and when should I use it?"

    print("\n" + "=" * 80)
    print("ROOT USER AGENT")
    print("=" * 80)

    user_agent = ROOTUserAgent(indexer.as_retriever(search_kwargs={"k": 5}))
    user_response = user_agent.ask(user_question)

    print(f"\nQuestion: {user_question}")
    print(f"\nAnswer:\n{user_response['answer']}")
    print(f"\n📚 Sources ({len(user_response['sources'])} documents):")
    for i, source in enumerate(user_response['sources'], 1):
        print(f"  [{i}] {source['title']} - {source['url']}")

    print("\n" + "=" * 80)
    print("ROOT DEVELOPER AGENT")
    print("=" * 80)

    developer_agent = ROOTDeveloperAgent(indexer.as_retriever(search_kwargs={"k": 7}))
    developer_response = developer_agent.ask(developer_question)

    print(f"\nQuestion: {developer_question}")
    print(f"\nAnswer:\n{developer_response['answer']}")
    print(f"\n📚 Sources ({len(developer_response['sources'])} documents):")
    for i, source in enumerate(developer_response['sources'], 1):
        print(f"  [{i}] {source['title']} - {source['url']}")

    print("\n" + "=" * 80)
    print("ROOT TEACHING AGENT")
    print("=" * 80)

    teaching_agent = ROOTTeachingAgent(indexer.as_retriever(search_kwargs={"k": 5}))
    teaching_response = teaching_agent.ask(teaching_question)

    print(f"\nQuestion: {teaching_question}")
    print(f"\nAnswer:\n{teaching_response['answer']}")
    print(f"\n📚 Sources ({len(teaching_response['sources'])} documents):")
    for i, source in enumerate(teaching_response['sources'], 1):
        print(f"  [{i}] {source['title']} - {source['url']}")


if __name__ == "__main__":
    main()
