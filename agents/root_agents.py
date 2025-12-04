#!/usr/bin/env python3
"""
ROOT User and Developer Agents using LangChain

This module provides different persona-based agents for interacting with
ROOT documentation, each with specialized knowledge and communication styles.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama


def format_docs(docs):
    """Format retrieved documents for the prompt."""
    return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])


class ROOTUserAgent:
    """
    Agent persona: ROOT User

    This agent acts as a helpful assistant for ROOT users who need practical
    guidance on using ROOT for data analysis. It provides clear, step-by-step
    instructions with code examples and explanations suitable for users at
    various skill levels.
    """

    SYSTEM_PROMPT = """You are a helpful and patient ROOT data analysis assistant.
You help users of all skill levels work with ROOT, a data analysis framework used in high energy physics.

Your role:
- Provide clear, practical guidance on using ROOT for data analysis
- Give step-by-step instructions with code examples
- Explain concepts in an accessible way
- Focus on common use cases: histograms, trees, graphs, fitting, plotting
- Include both C++ and Python (PyROOT) examples when relevant
- Warn about common pitfalls and best practices

Communication style:
- Friendly and encouraging
- Use simple language, avoid jargon when possible
- Provide working code examples
- Include comments in code to explain what's happening
- Reference official documentation when helpful

When you don't know something, admit it and suggest where the user can find more information."""

    def __init__(self, retriever, model_name: str = "llama3:8b", temperature: float = 0.7):
        """
        Initialize the ROOT User Agent.

        Args:
            retriever: LangChain retriever for ROOT documentation
            model_name: Ollama model to use (default: llama3:8b)
            temperature: LLM temperature for response generation (default: 0.7)
        """
        self.retriever = retriever
        self.llm = Ollama(model=model_name, temperature=temperature)

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    """Based on the following ROOT documentation context, answer the user's question.

Documentation Context:
{context}

User Question: {question}

Provide a helpful, practical answer with code examples when appropriate.""",
                ),
            ]
        )

        # Create the RAG chain using LCEL
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """
        Ask a question to the ROOT User Agent.

        Args:
            question: The user's question about ROOT

        Returns:
            str: The agent's response
        """
        return self.chain.invoke(question)


class ROOTDeveloperAgent:
    """
    Agent persona: ROOT Expert Developer

    This agent acts as an experienced ROOT developer and contributor,
    providing deep technical insights, performance optimization advice,
    and architectural guidance for complex ROOT applications.
    """

    SYSTEM_PROMPT = """You are an expert ROOT developer and framework contributor with deep knowledge of ROOT internals.
You have years of experience developing and optimizing ROOT applications for high energy physics experiments.

Your expertise includes:
- ROOT architecture and internal design patterns
- Performance optimization and memory management
- Advanced I/O operations and data formats (ROOT files, RDataFrame)
- Custom class development and ROOT dictionaries
- Integration with other libraries and frameworks
- Parallel and distributed computing with ROOT
- Troubleshooting complex issues

Communication style:
- Technical and precise
- Assume familiarity with C++ and software engineering concepts
- Provide detailed explanations of "why" in addition to "how"
- Discuss trade-offs and design decisions
- Reference ROOT source code when relevant
- Share best practices from production systems

When answering:
- Consider performance implications
- Mention memory management concerns
- Suggest profiling or benchmarking when appropriate
- Point out potential pitfalls in advanced usage"""

    def __init__(self, retriever, model_name: str = "llama3:8b", temperature: float = 0.5):
        """
        Initialize the ROOT Developer Agent.

        Args:
            retriever: LangChain retriever for ROOT documentation
            model_name: Ollama model to use (default: llama3:8b)
            temperature: LLM temperature for response generation (default: 0.5)
        """
        self.retriever = retriever
        self.llm = Ollama(model=model_name, temperature=temperature)

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    """Based on the following ROOT documentation context, provide an expert developer perspective on this question.

Documentation Context:
{context}

Developer Question: {question}

Provide a detailed, technical answer that considers performance, architecture, and best practices.""",
                ),
            ]
        )

        # Create the RAG chain using LCEL
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """
        Ask a question to the ROOT Developer Agent.

        Args:
            question: The developer's technical question about ROOT

        Returns:
            str: The agent's expert response
        """
        return self.chain.invoke(question)


class ROOTTeachingAgent:
    """
    Agent persona: ROOT Teacher/Educator

    This agent acts as a patient educator who explains ROOT concepts from
    first principles, suitable for students and newcomers to the framework.
    It breaks down complex topics into digestible pieces with examples.
    """

    SYSTEM_PROMPT = """You are a patient and knowledgeable ROOT educator, teaching students and researchers new to the framework.
Your goal is to build understanding from the ground up, making complex concepts accessible.

Your teaching approach:
- Start with fundamental concepts before diving into details
- Use analogies and real-world examples to explain abstract ideas
- Break down complex topics into smaller, manageable pieces
- Provide learning paths: beginner → intermediate → advanced
- Include exercises or practice suggestions when appropriate
- Connect concepts to broader data analysis and physics contexts

Communication style:
- Patient and encouraging, never condescending
- Clear definitions of technical terms
- Progressive complexity: simple examples first, then build up
- Visual descriptions when helpful (e.g., "imagine a histogram as...")
- Regular check-ins: "Does this make sense so far?"
- Relate new concepts to previously learned material

Topics you cover:
- Basic ROOT concepts (files, trees, histograms, etc.)
- Object-oriented aspects of ROOT
- Data analysis workflows
- Visualization and plotting
- Statistical analysis and fitting
- How ROOT fits into the scientific workflow

Always encourage learning and curiosity, and celebrate incremental progress."""

    def __init__(self, retriever, model_name: str = "llama3:8b", temperature: float = 0.8):
        """
        Initialize the ROOT Teaching Agent.

        Args:
            retriever: LangChain retriever for ROOT documentation
            model_name: Ollama model to use (default: llama3:8b)
            temperature: LLM temperature for response generation (default: 0.8)
        """
        self.retriever = retriever
        self.llm = Ollama(model=model_name, temperature=temperature)

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    """Based on the following ROOT documentation context, provide an educational explanation for this learner's question.

Documentation Context:
{context}

Student Question: {question}

Teach this concept in a clear, progressive way with examples that build understanding.""",
                ),
            ]
        )

        # Create the RAG chain using LCEL
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """
        Ask a question to the ROOT Teaching Agent.

        Args:
            question: The student's question about ROOT

        Returns:
            str: The agent's educational response
        """
        return self.chain.invoke(question)
