"""
Fitness Agent with RAG System (No Tools)
This module creates an AI agent with knowledge retrieval from PDF documents.
Uses Agno 2.0 Knowledge API with LanceDB for local vector storage.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
# Load environment variables
load_dotenv()


def create_fitness_agent_rag(
    rag_docs_path: str = "./rag_docs",
    recreate_knowledge: bool = False
) -> Agent:
    """
    Create a fitness agent with RAG capabilities (no tools).
    
    Args:
        rag_docs_path: Path to the folder containing PDF documents
        recreate_knowledge: Whether to recreate the knowledge base from scratch
    
    Returns:
        Configured Agent instance with RAG knowledge
    """
    
    print(f"📚 Setting up RAG knowledge base from: {rag_docs_path}")
    
    # Check if PDF documents exist
    pdf_path = Path(rag_docs_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Directory {rag_docs_path} not found. Please create it and add PDF files.")
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {rag_docs_path}. Please add PDF documents.")
    
    print(f"✅ Found {len(pdf_files)} PDF document(s)")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    # Set up LanceDB for local vector storage
    vector_db = LanceDb(
        table_name="fitness_documents",
        uri="tmp/lancedb",  # Local directory for LanceDB
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        #OpenAIEmbedder(
        #    id="text-embedding-3-small",
        #    api_key=os.getenv("OPENAI_API_KEY")
        #),
    )
    
    # Create Knowledge instance
    knowledge = Knowledge(
        name="Fitness Knowledge Base",
        description="Knowledge base containing fitness and health information from PDF documents",
        vector_db=vector_db,
    )
    
    # Load PDF documents into knowledge base
    print("\n📖 Loading PDF documents into knowledge base...")
    
    # Use asyncio to load content
    async def load_pdfs():
        for pdf_file in pdf_files:
            print(f"   Loading: {pdf_file.name}...")
            await knowledge.add_content_async(
                name=pdf_file.stem,  # Use filename without extension as name
                path=str(pdf_file.absolute()),
                metadata={
                    "filename": pdf_file.name,
                    "type": "fitness_document"
                }
            )
    
    # Run the async loading
    asyncio.run(load_pdfs())
    
    print("✅ Knowledge base loaded successfully!\n")
    
    # Create the agent with RAG knowledge
    agent = Agent(
        name="Fitness RAG Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=knowledge,
        search_knowledge=True,  # Enable knowledge search
        instructions=[
            "You are an expert fitness coach and health advisor.",
            "Use the knowledge base to provide accurate, evidence-based information.",
            "Always cite specific information from the documents when available.",
            "If the knowledge base doesn't contain information about a query, say so clearly.",
            "Be encouraging and supportive in your responses.",
            "Provide detailed, comprehensive answers based on the available knowledge.",
        ],
        markdown=True,
    )
    
    return agent


def main():
    """
    Main function to create and test the RAG fitness agent.
    """
    print("🏋️ Creating Fitness Agent with RAG System...\n")
    
    try:
        # Create the agent (set recreate_knowledge=True to rebuild the knowledge base)
        agent = create_fitness_agent_rag(recreate_knowledge=False)
        
        print("✅ Fitness RAG Agent created successfully!")
        print("\nAgent capabilities:")
        print("  - RAG system with fitness/health PDF documents")
        print("  - Knowledge-based question answering")
        print("  - GPT-4o-mini powered responses")
        print("  - LanceDB local vector storage")
        
        return agent
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Create the 'rag_docs' folder: mkdir -p rag_docs")
        print("2. Add your fitness/health PDF documents to the folder")
        print("3. Run this script again")
        return None
    except Exception as e:
        print(f"\n❌ Error creating agent: {e}")
        return None


if __name__ == "__main__":
    agent = main()
    
    if agent:
        # Test the agent with a sample query
        print("\n" + "="*60)
        print("Testing agent with a sample query...")
        print("="*60 + "\n")
        
        test_query = "What are the key principles of effective strength training?"
        print(f"Query: {test_query}\n")
        agent.print_response(test_query, stream=True)