#!/usr/bin/env python

"""
Example demonstrating how to use company AI proxies with the data generator.
This enables using the library in enterprise environments where AI API calls
are typically routed through internal proxy services.

⚠️ WARNING: The proxy configuration feature is experimental and has not been
thoroughly tested with actual enterprise proxy setups. You may need to adjust 
the implementation to work with your specific company proxy.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the syda package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig, ProxyConfig

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    # Simple schema for generating product data
    schema = {
        "product_id": "number",
        "product_name": "text",
        "description": "text",
        "price": "number",
        "category": "text",
        "in_stock": "boolean"
    }
    
    prompt = "Generate premium electronic products with detailed descriptions"
    
    # Example 1: Using OpenAI through a company proxy
    print("\n1. Using OpenAI with company proxy:")
    
    # Create a proxy configuration for your company's OpenAI proxy
    openai_proxy_config = ModelConfig(
        provider="openai",
        model_name="gpt-4", 
        temperature=0.7,
        proxy=ProxyConfig(
            base_url="https://ai-proxy.company.com/v1",  # Replace with your company's proxy URL
            headers={
                "X-Company-Auth": "your-internal-token",  # Replace with actual auth if needed
                "X-Project-ID": "synthetic-data"
            },
            # Query parameters will be appended to the URL: https://ai-proxy.company.com/v1?team=data-science&project=synthetic-data
            params={
                "team": "data-science",
                "project": "synthetic-data",
                "track_usage": True,  # Will be converted to string "True"
                "priority": 1          # Will be converted to string "1"
            }
        )
    )
    
    print("When configured with params, the actual URL used will be:")
    print("https://ai-proxy.company.com/v1?team=data-science&project=synthetic-data&track_usage=True&priority=1")
    
    # Create a generator that uses the proxy
    proxy_generator = SyntheticDataGenerator(model_config=openai_proxy_config)
    
    # Note: This will only work if your company's proxy is correctly configured
    # Uncomment the following lines to test with your actual proxy setup
    """
    proxy_data = proxy_generator.generate_data(
        schema_dict=schema,
        prompt=prompt,
        sample_size=2
    )
    print(proxy_data)
    """
    print("(Example code - requires actual proxy configuration to run)")
    
    # Example 2: Using Anthropic models through a proxy
    print("\n2. Using Anthropic with company proxy:")
    
    anthropic_proxy_config = ModelConfig(
        provider="anthropic",
        model_name="claude-3-haiku-20240307",
        temperature=0.5,
        proxy=ProxyConfig(
            base_url="https://ai-proxy.company.com/anthropic",  # Replace with your company's Anthropic proxy
            headers={"X-API-Source": "synthetic-data-generator"}
        )
    )
    
    # Create generator with Anthropic proxy
    anthropic_proxy_generator = SyntheticDataGenerator(model_config=anthropic_proxy_config)
    
    # Note: This will only work if your company's Anthropic proxy is configured
    print("(Example code - requires actual proxy configuration to run)")
    
    print("\n3. Using proxy with custom authentication:")
    
    # Example of more complex proxy setup with custom auth
    complex_proxy_config = ModelConfig(
        provider="openai",
        model_name="gpt-4-turbo",
        temperature=0.8,
        proxy=ProxyConfig(
            base_url="https://ai-proxy.company.com/openai",
            headers={
                "Authorization": f"Bearer {os.environ.get('COMPANY_PROXY_TOKEN', 'your-token-here')}",
                "X-Request-Source": "syda-library"
            },
            path_format="/proxy/llm/{provider}/completions"
        )
    )
    
    # Create generator with complex proxy setup
    complex_proxy_generator = SyntheticDataGenerator(model_config=complex_proxy_config)
    
    print("(Example code - requires actual proxy configuration to run)")
    
    print("\nNotes about proxy configuration:")
    print("- Replace the example URLs and tokens with your company's actual proxy settings")
    print("- Ensure you have the necessary permissions to access the proxy")
    print("- Check with your IT department for the correct headers and authentication method")
    print("- Consider storing proxy tokens in environment variables rather than hardcoding")

if __name__ == "__main__":
    main()
