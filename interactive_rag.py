#!/usr/bin/env python3
"""
Interactive RAG with LangChain Agents

This script provides an interactive command-line interface to ask questions
to the HEP RAG system using different agent personas.
"""

import sys
from agents import LangChainHEPIndexer, ROOTUserAgent, ROOTDeveloperAgent, ROOTTeachingAgent


def print_header():
    """Print the welcome header."""
    print("\n" + "=" * 80)
    print("🚀 HEP RAG Interactive Assistant with LangChain Agents")
    print("=" * 80)
    print("\nAvailable Agents:")
    print("  1. 🤝 User Agent     - Practical help for ROOT users")
    print("  2. 💻 Developer Agent - Expert technical guidance")
    print("  3. 📚 Teacher Agent   - Educational explanations")
    print("  4. 🔍 Direct Search   - Raw retrieval (no LLM)")
    print("\nCommands:")
    print("  /switch <1-4>  - Switch to a different agent")
    print("  /history       - Show conversation history")
    print("  /clear         - Clear screen")
    print("  /help          - Show this help")
    print("  /exit or /quit - Exit the program")
    print("=" * 80 + "\n")


def print_separator():
    """Print a visual separator."""
    print("\n" + "-" * 80 + "\n")


def format_sources(sources):
    """Format source documents for display."""
    if not sources:
        return ""
    
    result = "\n📚 Sources:\n"
    for i, source in enumerate(sources, 1):
        result += f"  [{i}] {source.get('title', 'Unknown')} - {source.get('url', 'No URL')}\n"
    return result


def select_agent(indexer):
    """Prompt user to select an agent."""
    while True:
        print("\nSelect an agent:")
        print("  1. 🤝 User Agent (practical help)")
        print("  2. 💻 Developer Agent (expert technical)")
        print("  3. 📚 Teacher Agent (educational)")
        print("  4. 🔍 Direct Search (no LLM)")
        
        choice = input("\nYour choice (1-4): ").strip()
        
        if choice == "1":
            return "user", ROOTUserAgent(indexer.as_retriever(search_kwargs={"k": 5}))
        elif choice == "2":
            return "developer", ROOTDeveloperAgent(indexer.as_retriever(search_kwargs={"k": 7}))
        elif choice == "3":
            return "teacher", ROOTTeachingAgent(indexer.as_retriever(search_kwargs={"k": 5}))
        elif choice == "4":
            return "search", indexer
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")


def handle_direct_search(indexer, query):
    """Handle direct search without LLM."""
    print("\n🔍 Searching documentation...")
    results = indexer.search(query, k=5)
    
    if not results:
        print("❌ No results found.")
        return
    
    print(f"\n✅ Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['title']}")
        print(f"    Score: {result['score']:.3f}")
        print(f"    URL: {result['url']}")
        print(f"    Content: {result['content'][:300]}...")
        print()


def main():
    """Main interactive loop."""
    # Initialize indexer
    print("🔧 Initializing LangChain indexer...")
    try:
        indexer = LangChainHEPIndexer(
            es_host="http://localhost:9200",
            index_name="root-documentation"
        )
        print("✅ Indexer initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing indexer: {e}")
        print("\nMake sure:")
        print("  • Elasticsearch is running on localhost:9200")
        print("  • ROOT documentation is indexed in 'root-documentation'")
        sys.exit(1)
    
    # Print header
    print_header()
    
    # Select initial agent
    agent_type, agent = select_agent(indexer)
    
    # Agent names for display
    agent_names = {
        "user": "🤝 User Agent",
        "developer": "💻 Developer Agent",
        "teacher": "📚 Teacher Agent",
        "search": "🔍 Direct Search"
    }
    
    # Conversation history
    history = []
    
    print(f"\n✅ Using {agent_names[agent_type]}")
    print("Type your question or use /help for commands\n")
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = input(f"{agent_names[agent_type]} > ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command in ["/exit", "/quit"]:
                    print("\n👋 Goodbye!")
                    break
                
                elif command == "/help":
                    print_header()
                    continue
                
                elif command == "/clear":
                    print("\033[2J\033[H")  # Clear screen
                    print_header()
                    print(f"\n✅ Using {agent_names[agent_type]}")
                    continue
                
                elif command == "/history":
                    if not history:
                        print("\n📋 No conversation history yet.")
                    else:
                        print("\n📋 Conversation History:")
                        print_separator()
                        for i, (q, a) in enumerate(history, 1):
                            print(f"[{i}] Q: {q}")
                            print(f"    A: {a[:200]}...")
                            print()
                    continue
                
                elif command.startswith("/switch"):
                    parts = command.split()
                    if len(parts) == 2 and parts[1] in ["1", "2", "3", "4"]:
                        choice = parts[1]
                        if choice == "1":
                            agent_type = "user"
                            agent = ROOTUserAgent(indexer.as_retriever(search_kwargs={"k": 5}))
                        elif choice == "2":
                            agent_type = "developer"
                            agent = ROOTDeveloperAgent(indexer.as_retriever(search_kwargs={"k": 7}))
                        elif choice == "3":
                            agent_type = "teacher"
                            agent = ROOTTeachingAgent(indexer.as_retriever(search_kwargs={"k": 5}))
                        elif choice == "4":
                            agent_type = "search"
                            agent = indexer
                        
                        print(f"\n✅ Switched to {agent_names[agent_type]}")
                    else:
                        print("\n❌ Usage: /switch <1-4>")
                    continue
                
                else:
                    print(f"\n❌ Unknown command: {user_input}")
                    print("Type /help for available commands")
                    continue
            
            # Process question
            print(f"\n💭 Processing your question...")
            
            try:
                if agent_type == "search":
                    # Direct search
                    handle_direct_search(agent, user_input)
                else:
                    # LLM agent
                    result = agent.ask(user_input)
                    
                    print_separator()
                    print(f"📝 Answer:\n")
                    print(result['answer'])
                    
                    # Show sources
                    if result.get('sources'):
                        print("\n" + "=" * 80)
                        print("📚 SOURCES:")
                        print("=" * 80)
                        for i, source in enumerate(result['sources'], 1):
                            print(f"\n[{i}] {source['title']}")
                            print(f"    🔗 {source['url']}")
                            print(f"    📄 {source['content']}")
                    
                    print_separator()
                    
                    # Store in history
                    history.append((user_input, result['answer']))
                
            except Exception as e:
                print(f"\n❌ Error processing question: {e}")
                print("\nMake sure:")
                if agent_type != "search":
                    print("  • Ollama is running (ollama serve)")
                    print("  • llama3:8b model is available (ollama pull llama3:8b)")
                print("  • Elasticsearch is accessible")
                print("  • Documentation is indexed")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
