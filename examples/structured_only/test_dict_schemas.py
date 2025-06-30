#!/usr/bin/env python
"""
Example of using the enhanced SyntheticDataGenerator to automatically handle
multiple related dictionary schemas with foreign key relationships.

This example demonstrates:
1. Defining a set of related schemas as dictionaries
2. Using generate_for_schemas to automatically handle:
   - Dependency resolution between schemas
   - Generation order (parents before children)
   - Foreign key constraints
   - Column-specific generators
3. All without manually managing the process
"""

import sys
import os
import random
from dotenv import load_dotenv
import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the synthetic data generator
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig


def main():
    """Demonstrate automatic generation of related data with proper foreign key handling."""
    
    # Create a generator instance with appropriate max_tokens setting
    model_config = ModelConfig(
        provider="anthropic",
        #model_name="gpt-4",  # Default model
        model_name="claude-3-5-haiku-20241022",
        temperature=0.7,
        max_tokens=8192,  # Using higher max_tokens value for more complete responses
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Define output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "output", "test_dict_schemas", 
        "ecommerce"
    )
    
    # Define schema dictionaries for an e-commerce system with descriptions
    schemas = {
        # Customer schema with table and column descriptions
        'Customer': {
            # Define schema with additional metadata
            '__table_description__': 'Registered users of the e-commerce platform who can place orders',
            'id': {'type': 'number', 'description': 'Unique identifier for the customer'},
            'name': {'type': 'text', 'description': 'Full name of the customer'},
            'email': {'type': 'email', 'description': 'Customer email address used for communication'},
            'signup_date': {'type': 'date', 'description': 'Date when the customer registered'},
            'loyalty_tier': {'type': 'text', 'description': 'Customer loyalty program level (Bronze, Silver, Gold, Platinum)'}
        },
        
        # Product schema with table and column descriptions
        'Product': {
            '__table_description__': 'Products available for purchase in the e-commerce store',
            'id': {'type': 'number', 'description': 'Unique identifier for the product'},
            'name': {'type': 'text', 'description': 'Name of the product as displayed to customers'},
            'category': {'type': 'text', 'description': 'Product category for classification and filtering'},
            'price': {'type': 'number', 'description': 'Current price of the product in USD'},
            'description': {'type': 'text', 'description': 'Detailed description of the product features and benefits'},
            'in_stock': {'type': 'boolean', 'description': 'Whether the product is currently available for purchase'}
        },
        
        # Order schema with table and column descriptions
        'Order': {
            # Define schema with additional metadata
            '__table_description__': 'Customer orders for products, including order status and total amount',
            '__foreign_keys__': {
                'customer_id': ['Customer', 'id']  # Order.customer_id references Customer.id
            },
            
            # Define columns
            'id': {'type': 'number', 'description': 'Unique order identifier', 'primary_key': True},
            'customer_id': {'type': 'foreign_key', 'description': 'Reference to the customer who placed the order'},
            'order_date': {'type': 'date', 'description': 'Date when the order was placed'},
            'status': {'type': 'text', 'description': 'Current status of the order (Pending, Processing, Shipped, Delivered, Cancelled)'},
            'total_amount': {'type': 'number', 'description': 'Total amount of the order in USD'},
            'shipping_address': {'type': 'text', 'description': 'Address where the order should be delivered'}
        },
        
        # OrderItem schema with table and column descriptions
        'OrderItem': {
            '__table_description__': 'Individual line items within an order, representing specific products',
            '__foreign_keys__': {
                'order_id': ['Order', 'id'],       # OrderItem.order_id references Order.id
                'product_id': ['Product', 'id']    # OrderItem.product_id references Product.id
            },
            
            # Define columns
            'id': {'type': 'number', 'description': 'Unique identifier for the order item'},
            'order_id': {'type': 'foreign_key', 'description': 'Reference to the parent order'},
            'product_id': {'type': 'foreign_key', 'description': 'Reference to the product being ordered'},
            'quantity': {'type': 'number', 'description': 'Number of units of the product ordered'},
            'unit_price': {'type': 'number', 'description': 'Price per unit at the time of order, may differ from current product price'}
        }
    }
    
    # Define foreign key relationships
    foreign_keys = {
        'Order': {
            'customer_id': ('Customer', 'id')  # Order.customer_id references Customer.id
        },
        'OrderItem': {
            'order_id': ('Order', 'id'),       # OrderItem.order_id references Order.id
            'product_id': ('Product', 'id')    # OrderItem.product_id references Product.id
        }
    }
    
    # Define custom prompts for each schema (optional)
    prompts = {
        "Customer": """
        Generate diverse customers for an e-commerce platform.
        Include various loyalty tiers (Bronze, Silver, Gold, Platinum)
        and realistic signup dates within the last 3 years.
        """,
        
        "Product": """
        Generate diverse products for an e-commerce store.
        Include various categories (Electronics, Clothing, Home, Books, etc.)
        with realistic prices and descriptions.
        """,
        
        "Order": """
        Generate realistic orders with appropriate dates and statuses
        (Pending, Processing, Shipped, Delivered, Cancelled).
        Total amounts should reflect typical e-commerce purchases.
        """
    }
    
    # Define sample sizes for each schema (optional)
    sample_sizes = {
        "Customer": 10,       # Base entities
        "Product": 15,        # Product catalog
        "Order": 25,          # ~2-3 orders per customer
        "OrderItem": 50,      # ~2 items per order
    }
    
    # Define custom generators for specific schema columns
    # Define custom generators for specific schema fields
    #
    # NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
    # based on field names, types, and descriptions. Custom generators give you precise control 
    # for fields where you need specific distributions or formatting.
    #
    # This example demonstrates using dictionary-based schemas with foreign key relationships
    # and just a few strategic custom generators:
    #
    custom_generators = {
        "Customer": {
            # Ensure loyalty tiers match your specific business structure
            "loyalty_tier": lambda row, col: random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
        },
        "Product": {
            # Create a strategic product category distribution
            "category": lambda row, col: random.choice([
                "Electronics", "Clothing", "Home & Kitchen", "Books", 
                "Beauty", "Sports", "Toys"
            ])
        },
        "Order": {
            # Create a realistic distribution of order statuses
            "status": lambda row, col: random.choices(
                ["Pending", "Processing", "Shipped", "Delivered", "Cancelled"],
                weights=[0.1, 0.15, 0.2, 0.5, 0.05]  # More likely to be delivered
            )[0]
        }
    }
    
    print("\n🔄 Generating related data for E-commerce system...")
    print("  The system will automatically determine the right generation order")
    print("  and set up foreign key relationships")
    print("  with custom generators for specific columns\n")
    
    # Generate data using the unified generate_for_schemas method
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=output_dir,
        custom_generators=custom_generators
    )
    
    # Print summary
    print("\n✅ Data generation complete!")
    for schema_name, df in results.items():
        print(f"  - {schema_name}: {len(df)} records")
    
    print(f"\nData files saved to directory: {output_dir}/")
    
    # Show samples of data with custom generators
    print("\n📊 Sample data with custom generators:")
    
    print("\nCustomer loyalty tiers (custom generator):")
    loyalty_counts = results["Customer"]["loyalty_tier"].value_counts().to_dict()
    for tier, count in loyalty_counts.items():
        print(f"  - {tier}: {count} customers")
    
    print("\nProduct categories (custom generator):")
    category_counts = results["Product"]["category"].value_counts().to_dict()
    for category, count in category_counts.items():
        print(f"  - {category}: {count} products")
    
    print("\nProduct prices (custom generator):")
    prices = results["Product"]["price"].tolist()
    print(f"  - Price range: ${min(prices):.2f} to ${max(prices):.2f}")
    print(f"  - Average price: ${sum(prices)/len(prices):.2f}")
    
    print("\n🔍 Verifying referential integrity:")
    
    # Check Orders → Customers
    order_customer_ids = set(results["Order"]["customer_id"].tolist())
    valid_customer_ids = set(results["Customer"]["id"].tolist())
    if order_customer_ids.issubset(valid_customer_ids):
        print("  ✅ All Order.customer_id values reference valid Customers")
    else:
        print("  ❌ Invalid Order.customer_id references detected")
    
    # Check OrderItems → Orders
    order_item_order_ids = set(results["OrderItem"]["order_id"].tolist())
    valid_order_ids = set(results["Order"]["id"].tolist())
    if order_item_order_ids.issubset(valid_order_ids):
        print("  ✅ All OrderItem.order_id values reference valid Orders")
    else:
        print("  ❌ Invalid OrderItem.order_id references detected")
    
    # Check OrderItems → Products
    order_item_product_ids = set(results["OrderItem"]["product_id"].tolist())
    valid_product_ids = set(results["Product"]["id"].tolist())
    if order_item_product_ids.issubset(valid_product_ids):
        print("  ✅ All OrderItem.product_id values reference valid Products")
    else:
        print("  ❌ Invalid OrderItem.product_id references detected")


if __name__ == "__main__":
    main()
