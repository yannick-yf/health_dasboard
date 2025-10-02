"""
Basic Agent Example - Where we ask question on body fat computation
It is clearly showcasing that with gpt-4o-mini models it is not working
"""

from textwrap import dedent

import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Load environment variables
load_dotenv()

# Create our News Reporter with a fun personality
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=dedent("""\
        You are an expert fitness coach and health advisor.
        Provide evidence-based advice about fitness, nutrition, and health.
        When users provide body measurements, use the calculate_body_fat tool.
        Always be encouraging and supportive in your responses.
        If asked about medical conditions, remind users to consult healthcare professionals.
    """),
    markdown=True,
)

# Example usage - Testing tools usecase without the tools being created
# agent.print_response(
#     "Can you calculate my body fat? I'm a male, 185cm tall, weighing 71kg, age 33, with a 39cm neck and 80cm waist? Provide function and details of your calculation.", stream=True
# )

# Testing RAG usecase without the RAG setup
agent.print_response(
    "Who are the authors and What is the purpose of the document 'Non-exercise activity thermogenesis (NEAT): a component of total daily energy expenditure'", stream=True
)


