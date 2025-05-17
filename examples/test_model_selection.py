#!/usr/bin/env python

"""
Example demonstrating how to use different AI models and configure model settings
for synthetic data generation.
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd

# Add the parent directory to the path so we can import the syda package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    # Define a simple schema for testing
    schema = {
        "product_id": "number",
        "product_name": "text",
        "description": "text",
        "price": "number",
        "category": "text",
        "in_stock": "boolean"
    }
    
    prompt = "Generate product data for an electronics store with realistic prices and descriptions."
    
    print("\n1. Default configuration (OpenAI GPT-4 with max_tokens=4000):")
    default_generator = SyntheticDataGenerator()
    default_data = default_generator.generate_data(
        schema_dict=schema,
        prompt=prompt,
        sample_size=3
    )
    print(f"Using default max_tokens: 4000")
    print(default_data)
    
    print("\n2. Using GPT-3.5 Turbo with explicit max_tokens:")
    gpt35_generator = SyntheticDataGenerator(
        model_config={
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.9,
            "max_tokens": 4000,  # Same as default, explicitly set
            "seed": 42  # For reproducible results
        }
    )
    gpt35_data = gpt35_generator.generate_data(
        schema_dict=schema,
        prompt=prompt,
        sample_size=3
    )
    print(gpt35_data)
    
    # Only run this example if ANTHROPIC_API_KEY is set
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n3. Using Claude with ModelConfig:")
        claude_config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            temperature=0.5,
            max_tokens=2000
        )
        claude_generator = SyntheticDataGenerator(model_config=claude_config)
        claude_data = claude_generator.generate_data(
            schema_dict=schema,
            prompt=prompt,
            sample_size=3
        )
        print(claude_data)
    else:
        print("\n3. Claude example skipped - ANTHROPIC_API_KEY not set")

    print("\n4. Complex configuration with all parameters:")
    complex_config = ModelConfig(
        provider="openai",
        model_name="gpt-4-turbo",
        temperature=0.8,
        max_tokens=1000,
        top_p=0.95,
        seed=123456
    )
    complex_generator = SyntheticDataGenerator(model_config=complex_config)
    complex_data = complex_generator.generate_data(
        schema_dict=schema,
        prompt=prompt,
        sample_size=3
    )
    print(complex_data)
    
    print("\nDifferent models may produce different quality or style of data!")


if __name__ == "__main__":
    main()
