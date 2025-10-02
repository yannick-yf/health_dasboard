"""
Query Script for Simple Fitness Agent
Works without RAG/vector database dependencies.
"""

import sys
from fitness_agent_simple import create_fitness_agent


def query_agent(question: str, debug: bool = True):
    """
    Send a query to the fitness agent and get a response.
    
    Args:
        question: The question to ask the agent
        debug: Whether to show detailed agent reasoning and tool calls
    
    Returns:
        Agent response
    """
    print("🏋️ Initializing Fitness Agent...")
    
    # Create the agent
    agent = create_fitness_agent()
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Query the agent with debug mode
    if debug:
        print("🔍 Agent Reasoning & Tool Calls:")
        print("-" * 60)
        response = agent.print_response(question, stream=True)
    else:
        print("🤖 Agent Response:\n")
        response = agent.run(question, stream=False)
        print(response.content)
    
    return response


def interactive_mode(debug: bool = True):
    """
    Run the agent in interactive mode for continuous conversation.
    
    Args:
        debug: Whether to show detailed agent reasoning and tool calls
    """
    print("🏋️ Starting Fitness Agent Interactive Mode")
    print(f"Debug mode: {'ON' if debug else 'OFF'} (type 'debug' to toggle)")
    print("Type 'exit' or 'quit' to end the session\n")
    
    # Create the agent once
    agent = create_fitness_agent()
    
    print("✅ Agent ready! Ask me anything about fitness, health, or body composition.\n")
    
    while True:
        try:
            # Get user input
            question = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thanks for using Fitness Agent! Stay healthy!")
                break
            
            # Toggle debug mode
            if question.lower() == 'debug':
                debug = not debug
                print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
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


if __name__ == "__main__":
    main()