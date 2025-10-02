"""
Simple Fitness Agent with only body fat calculator
"""

import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Load environment variables
load_dotenv()


def calculate_body_fat(
    neck_cm: float,
    waist_cm: float,
    height_cm: float,
    age: int,
    weight_kg: float,
    gender: str = "male"
) -> str:
    """
    Calculate body fat percentage and fat mass using U.S. Navy Body Fat Formula.
    
    Args:
        neck_cm: Neck circumference in centimeters
        waist_cm: Waist circumference in centimeters
        height_cm: Height in centimeters
        age: Age in years
        weight_kg: Weight in kilograms
        gender: 'male' or 'female'
    
    Returns:
        String with body fat calculation results
    """
    try:
        import math
        
        if gender.lower() == "male":
            # U.S. Navy formula for males
            # BF% = 495 / (1.0324 - 0.19077 * log10(waist - neck) + 0.15456 * log10(height)) - 450
            body_fat_pct = (495 / (1.0324 - 0.19077 * math.log10(waist_cm - neck_cm) + 
                           0.15456 * math.log10(height_cm))) - 450
        else:
            # U.S. Navy formula for females (requires hip measurement, using simplified version)
            # BF% = 495 / (1.29579 - 0.35004 * log10(waist + hip - neck) + 0.22100 * log10(height)) - 450
            # Since we don't have hip measurement, we'll use a modified formula
            body_fat_pct = (495 / (1.29579 - 0.35004 * math.log10(waist_cm - neck_cm) + 
                           0.22100 * math.log10(height_cm))) - 450
        
        # Ensure reasonable bounds
        body_fat_pct = max(3.0, min(body_fat_pct, 50.0))
        
        # Calculate fat mass
        fat_mass_kg = (body_fat_pct / 100) * weight_kg
        lean_mass_kg = weight_kg - fat_mass_kg
        
        # Calculate BMI for additional context
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        result = f"""
            Body Composition Analysis (U.S. Navy Method):
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Body Fat Percentage: {round(body_fat_pct, 1)}%
            Fat Mass: {round(fat_mass_kg, 2)} kg
            Lean Mass: {round(lean_mass_kg, 2)} kg
            Total Weight: {weight_kg} kg
            BMI: {round(bmi, 1)}

            Measurements Used:
            • Height: {height_cm} cm
            • Neck: {neck_cm} cm
            • Waist: {waist_cm} cm
            • Age: {age} years
            • Gender: {gender}

            Body Fat Category (Male):
            • 2-5%: Essential fat
            • 6-13%: Athletic
            • 14-17%: Fitness
            • 18-24%: Acceptable
            • 25%+: Obese

            Your result: {round(body_fat_pct, 1)}% - {"Athletic" if body_fat_pct < 14 else "Fitness" if body_fat_pct < 18 else "Acceptable" if body_fat_pct < 25 else "Obese"}
            """
        return result
        
    except Exception as e:
        return f"Error calculating body fat: {str(e)}. Please check that waist > neck and all measurements are positive."


def create_fitness_agent() -> Agent:
    """
    Create a simple fitness agent with body composition tool.
    
    Returns:
        Configured Agent instance
    """
    
    # Create the agent
    agent = Agent(
        name="Fitness Coach Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "You are an expert fitness coach and health advisor.",
            "Provide evidence-based advice about fitness, nutrition, and health.",
            "When users provide body measurements, use the calculate_body_fat tool.",
            "Always be encouraging and supportive in your responses.",
            "If asked about medical conditions, remind users to consult healthcare professionals.",
            "When using the body fat calculator, extract all required parameters: neck_cm, waist_cm, height_cm, age, weight_kg, and gender.",
        ],
        tools=[calculate_body_fat],
        markdown=True,
    )
    
    return agent


def main():
    """
    Main function to create and test the fitness agent.
    """
    print("🏋️ Creating Simple Fitness Agent...")
    
    # Create the agent
    agent = create_fitness_agent()
    
    print("✅ Fitness Agent created successfully!")
    print("\nAgent capabilities:")
    print("  - Body fat calculation tool")
    print("  - GPT-4o-mini powered responses")
    print("  - Fitness and health advice")
    
    return agent


if __name__ == "__main__":
    agent = main()
    
    # Test the agent with a simple query
    print("\n" + "="*50)
    print("Testing agent with a sample query...")
    print("="*50 + "\n")
    
    response = agent.run(
        "Can you calculate my body fat? I'm a male, 180cm tall, weighing 80kg, "
        "age 30, with a 38cm neck and 85cm waist."
    )
    print(response.content)