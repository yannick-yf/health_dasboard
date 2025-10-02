"""
Query Script for Fitness RAG Agent
This module provides an interface to interact with the RAG-enabled fitness agent.
"""

import sys
from fitness_agent_rag import create_fitness_agent_rag


def query_agent(question: str, debug: bool = True):
    """
    Send a query to the fitness RAG agent and get a response.
    
    Args:
        question: The question to ask the agent
        debug: Whether to show detailed agent reasoning
    
    Returns:
        Agent response
    """
    print("🏋️ Initializing Fitness RAG Agent...")
    print("📚 Loading knowledge base...\n")
    
    try:
        # Create the agent
        agent = create_fitness_agent_rag()
        
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Query the agent with debug mode
        if debug:
            print("🔍 Agent Response with Knowledge Search:")
            print("-" * 60)
            response = agent.print_response(question, stream=True)
        else:
            print("🤖 Agent Response:\n")
            response = agent.run(question, stream=False)
            print(response.content)
        
        return response
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease add PDF documents to the 'rag_docs' folder and try again.")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def interactive_mode(debug: bool = True):
    """
    Run the agent in interactive mode for continuous conversation.
    
    Args:
        debug: Whether to show detailed agent reasoning
    """
    print("🏋️ Starting Fitness RAG Agent Interactive Mode")
    print(f"Debug mode: {'ON' if debug else 'OFF'} (type 'debug' to toggle)")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'reload' to reload the knowledge base\n")
    
    try:
        # Create the agent once
        agent = create_fitness_agent_rag()
        
        print("\n✅ Agent ready! Ask me questions about fitness and health.\n")
        print("💡 Tip: The agent will search the PDF documents to answer your questions.\n")
        
        while True:
            try:
                # Get user input
                question = input("\n💬 You: ").strip()
                
                # Check for exit commands
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thanks for using Fitness RAG Agent! Stay healthy!")
                    break
                
                # Toggle debug mode
                if question.lower() == 'debug':
                    debug = not debug
                    print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
                    continue
                
                # Reload knowledge base
                if question.lower() == 'reload':
                    print("\n🔄 Reloading knowledge base...")
                    agent = create_fitness_agent_rag(recreate_knowledge=True)
                    print("✅ Knowledge base reloaded!\n")
                    continue
                
                if not question:
                    continue
                
                # Query the agent
                print("\n🤖 Agent: ")
                print("-" * 60)
                
                if debug:
                    # Show full reasoning with streaming
                    agent.print_response(question, stream=True)
                else:
                    # Just show final answer
                    response = agent.run(question, stream=False)
                    print(response.content)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again.")
                
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Create the 'rag_docs' folder: mkdir -p rag_docs")
        print("2. Add your fitness/health PDF documents to the folder")
        print("3. Run this script again")
    except Exception as e:
        print(f"\n❌ Error initializing agent: {e}")


def main():
    """
    Main function to handle command-line queries.
    """
    # Check for debug flag
    debug = True
    args = sys.argv[1:]
    
    if '--no-debug' in args:
        debug = False
        args.remove('--no-debug')
    
    if '--debug' in args:
        debug = True
        args.remove('--debug')
    
    if len(args) > 0:
        # If arguments provided, treat as a single query
        question = " ".join(args)
        query_agent(question, debug=debug)
    else:
        # Otherwise, run in interactive mode
        interactive_mode(debug=debug)


# Example queries for testing
EXAMPLE_QUERIES = [
    "Who are the authors What is the purpose of the document 'Non-exercise activity thermogenesis (NEAT): a component of total daily energy expenditure'"
    "What are the main principles of strength training?",
    "How should I structure my workout routine?",
    "What role does nutrition play in fitness?",
    "What are the benefits of cardiovascular exercise?",
    "How can I prevent injuries during training?",
]


if __name__ == "__main__":
    # Uncomment to test with example queries
    # print("Testing with example queries:\n")
    # for i, query in enumerate(EXAMPLE_QUERIES, 1):
    #     print(f"\n{'='*60}")
    #     print(f"Example {i}/{len(EXAMPLE_QUERIES)}")
    #     print(f"{'='*60}")
    #     query_agent(query, debug=True)
    #     if i < len(EXAMPLE_QUERIES):
    #         input("\nPress Enter to continue to next example...")
    
    # Run main interactive/CLI mode
    main()