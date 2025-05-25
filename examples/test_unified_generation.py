#!/usr/bin/env python3
"""
Example demonstrating the unified data generation system.

This example shows how to generate both structured data and template-based documents
using the SyntheticDataGenerator. It demonstrates:

1. Defining both structured and template schemas
2. Foreign key relationships between schemas
3. Template schemas referencing structured data
4. Generating and saving both structured data and documents

This uses Claude Haiku for generation, similar to other examples.
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the parent directory to the path so we can import the syda package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file if it exists
load_dotenv()

# Import the synthetic data generator
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Define the output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'unified_generation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define schema definitions for both structured data and templates
schemas = {
    # Structured schemas
    'Customer': {
        'id': 'number',
        'name': 'text',
        'email': 'email',
        'address': 'text',
        'phone': 'phone'
    },
    
    'Product': {
        'id': 'number',
        'name': 'text',
        'description': 'text',
        'price': 'number',
        'category': 'text'
    },
    
    'Order': {
        '__foreign_keys__': {
            'customer_id': ['Customer', 'id']
        },
        'id': 'number',
        'customer_id': 'foreign_key',
        'order_date': 'date',
        'status': 'text',
        'total_amount': 'number'
    },
    
    'OrderItem': {
        '__foreign_keys__': {
            'order_id': ['Order', 'id'],
            'product_id': ['Product', 'id']
        },
        'id': 'number',
        'order_id': 'foreign_key',
        'product_id': 'foreign_key',
        'quantity': 'number',
        'unit_price': 'number',
        'subtotal': 'number'
    },
    
    # Template schema for invoices
    'InvoiceTemplate': {
        '__template__': {
            'source': 'examples/templates/invoice_template.txt'
        },
        # Foreign key relationships to structured data
        '__foreign_keys__': {
            'customer_name': ['Customer', 'name'],
            'customer_address': ['Customer', 'address'],
            'customer_email': ['Customer', 'email']
        },
        # Field definitions for template placeholders
        'invoice_number': 'text',
        'invoice_date': 'date',
        'due_date': 'date',
        'customer_name': 'text',
        'customer_address': 'text',
        'customer_email': 'email',
        'items': 'text',
        'subtotal': 'number',
        'tax_rate': 'number',
        'tax_amount': 'number',
        'total_amount': 'number'
    }
}

# Define custom prompts for each schema
prompts = {
    'Customer': 'Generate realistic customer data for an e-commerce company',
    'Product': 'Generate technology product data',
    'Order': 'Generate order data for an e-commerce company',
    'OrderItem': 'Generate order item data for technology products',
    'InvoiceTemplate': 'Generate invoice data for customers'
}

# Define sample sizes for each schema
sample_sizes = {
    'Customer': 5,
    'Product': 10,
    'Order': 8,
    'OrderItem': 15,
    'InvoiceTemplate': 3
}

# Define custom generators for specific fields
def order_status_generator(row, col):
    import random
    return random.choice(['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled'])

def tax_rate_generator(row, col):
    return 8.5  # Fixed tax rate

# Create custom generators dictionary
custom_generators = {
    'Order': {
        'status': order_status_generator
    },
    'InvoiceTemplate': {
        'tax_rate': tax_rate_generator
    }
}

def main():
    """Run the example."""
    print("Unified Data Generation Example")
    print("=" * 40)
    
    # Initialize the generator with Claude Haiku
    config = ModelConfig(provider="anthropic", model="claude-3-haiku-20240307")
    generator = SyntheticDataGenerator(model_config=config)
    
    # Register any global custom generators if needed
    generator.register_generator('phone', lambda row, col: f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}")
    
    print(f"\nGenerating data for {len(schemas)} schemas...")
    
    # Generate data for all schemas - structured and template-based
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=OUTPUT_DIR,
        custom_generators=custom_generators,
        output_format='json'  # Save structured data as JSON
    )
    
    # Print summary of results
    print("\nGeneration Complete!")
    print("=" * 40)
    
    for schema_name, data in results.items():
        if hasattr(data, 'shape'):  # It's a DataFrame (structured data)
            print(f"{schema_name}: {len(data)} records generated")
        else:  # It's a list of documents (template data)
            print(f"{schema_name}: {len(data)} documents generated")
    
    print("\nOutput saved to:", OUTPUT_DIR)
    
    # Display sample of generated data
    print("\nSample of Customer data:")
    print(results['Customer'].head(2))
    
    print("\nSample of Invoice Template (first document):")
    if len(results['InvoiceTemplate']) > 0:
        print(results['InvoiceTemplate'][0][:300] + "...")
    
    print("\nDone!")

if __name__ == "__main__":
    import random  # For custom generators
    main()
