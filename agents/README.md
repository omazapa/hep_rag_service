# ROOT Agents - LangChain-based Personas

This directory contains LangChain-based agents with different personas for interacting with ROOT documentation.

## 🎭 Available Agents

### 1. **ROOTUserAgent** - Practical User Assistant
- **Persona**: Helpful assistant for ROOT users
- **Target Audience**: Users at all skill levels
- **Communication Style**: Friendly, clear, practical
- **Focus**: 
  - Common use cases (histograms, trees, graphs, fitting)
  - Working code examples (C++ and PyROOT)
  - Step-by-step instructions
  - Best practices and common pitfalls

### 2. **ROOTDeveloperAgent** - Expert Technical Advisor
- **Persona**: Experienced ROOT core developer
- **Target Audience**: Advanced users and contributors
- **Communication Style**: Technical, precise, in-depth
- **Focus**:
  - Architecture and internals
  - Performance optimization
  - Design patterns
  - Thread safety and memory management
  - Advanced features (I/O, RDataFrame, CLING)

### 3. **ROOTTeachingAgent** - Patient Tutor
- **Persona**: Educational mentor
- **Target Audience**: Newcomers to ROOT/HEP
- **Communication Style**: Pedagogical, encouraging
- **Focus**:
  - Fundamental concepts
  - Progressive learning
  - Analogies and examples
  - Practice exercises
  - Building understanding step-by-step

## 📦 Installation

Install required dependencies:

```bash
pip install langchain langchain-elasticsearch langchain-huggingface langchain-community
```

## 🚀 Usage

### Basic Example

```python
from agents import LangChainHEPIndexer, ROOTUserAgent, ROOTDeveloperAgent

# Initialize indexer
indexer = LangChainHEPIndexer(
    es_host="http://localhost:9200",
    index_name="root-documentation"
)

# User Agent - Practical help
user_agent = ROOTUserAgent(indexer)
response = user_agent.ask("How do I create a histogram?")
print(response['answer'])

# Developer Agent - Technical details
dev_agent = ROOTDeveloperAgent(indexer)
response = dev_agent.ask("What are the performance implications of TTree vs RDataFrame?")
print(response['answer'])
```

### Using Custom LLM

```python
# You can pass any LangChain-compatible LLM
# For example, using a different Ollama model:
from langchain_community.llms import Ollama

llm = Ollama(model="mistral", temperature=0.7)

user_agent = ROOTUserAgent(indexer, llm=llm)
response = user_agent.ask("How do I fit a Gaussian to my histogram?")
```

### Complete Example

Run the example script:

```bash
python agents/example_usage.py
```

## 🎯 Agent Characteristics

| Agent | Temperature | Context Chunks (k) | Prompt Focus |
|-------|-------------|-------------------|--------------|
| ROOTUserAgent | 0.7 | 5 | Practical examples, clear instructions |
| ROOTDeveloperAgent | 0.5 | 7 | Technical depth, architecture |
| ROOTTeachingAgent | 0.8 | 5 | Pedagogical, progressive learning |

## 🔧 Customization

### Modify Prompts

Each agent has customizable prompts:

```python
# Customize system prompt
ROOTUserAgent.SYSTEM_PROMPT = "Your custom system prompt..."

# Or create a subclass
class MyCustomAgent(ROOTUserAgent):
    SYSTEM_PROMPT = """
    You are a specialized ROOT assistant for experimental physics...
    """
```

### Adjust Retrieval

```python
# Retrieve more context for complex questions
response = developer_agent.ask("Complex question...", k=10)
```

## 📚 Architecture

```
agents/
├── __init__.py              # Package exports
├── langchain_indexer.py     # LangChain wrapper for Elasticsearch
├── root_agents.py           # Agent personas and prompts
├── example_usage.py         # Usage examples
└── README.md               # This file
```

## 🎨 Prompt Engineering

Each agent uses a two-part prompt:

1. **System Prompt**: Defines persona, role, and communication style
2. **User Prompt Template**: Structures the question-answering format

The prompts are designed to:
- Maintain consistent persona
- Leverage documentation context
- Provide actionable, accurate answers
- Include relevant code examples
- Reference sources appropriately

## 💡 Best Practices

1. **Choose the right agent** for your audience
2. **Index your documentation** before using agents
3. **Use higher k values** for complex questions
4. **Adjust temperature** based on creativity vs precision needs
5. **Provide clear questions** for better retrieval

## 🔍 Example Questions by Agent

### ROOTUserAgent
- "How do I create and fill a TH1F histogram?"
- "What's the difference between TTree and TNtuple?"
- "How can I save my histogram to a ROOT file?"

### ROOTDeveloperAgent
- "What are the performance implications of TTree basket sizes?"
- "How does ROOT's I/O compression work internally?"
- "What thread-safety guarantees does RDataFrame provide?"

### ROOTTeachingAgent
- "What is a histogram and why is it useful?"
- "Can you explain what a TTree is from scratch?"
- "What are the basic concepts I need to start with ROOT?"

## 🤝 Contributing

To add new agent personas:

1. Create a new class in `root_agents.py`
2. Define `SYSTEM_PROMPT` and prompt template
3. Implement the `ask()` method
4. Add to `__init__.py` exports
5. Add examples to this README

## 📄 License

MIT
