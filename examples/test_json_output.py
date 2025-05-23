#!/usr/bin/env python
"""
Example of using the SyntheticDataGenerator with JSON output format.

This example demonstrates:
1. Loading schemas from YAML files
2. Generating data based on these schemas
3. Saving the output in JSON format instead of CSV
4. Foreign key relationships defined within the schema files
5. Custom generators for specific fields
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
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig


def main():
    """Demonstrate synthetic data generation with JSON output format."""
    
    # Create a generator instance with appropriate max_tokens setting
    model_config = ModelConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        temperature=0.7,
        max_tokens=8192,
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Define output directory
    output_dir = "inventory_data_json"
    
    # Define paths to schema files
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_files/yaml_only")
    
    # Dictionary mapping schema names to their YAML file paths
    schemas = {
        "Supplier": os.path.join(schema_dir, "supplier.yml"),
        "Category": os.path.join(schema_dir, "category.yml"),
        "Product": os.path.join(schema_dir, "product.yml"),
        "Inventory": os.path.join(schema_dir, "inventory.yml")
    }
    
    # Define custom prompts for each schema (optional)
    prompts = {
        "Supplier": """
        Generate diverse suppliers for an inventory management system.
        Include manufacturers, distributors, and direct wholesalers.
        Create a range of payment terms and active statuses.
        """,
        
        "Category": """
        Generate product categories for an inventory system.
        Include top-level categories and some subcategories.
        Create descriptive names and detailed descriptions.
        """,
        
        "Product": """
        Generate diverse products for an inventory system.
        Include various categories, price points, and availability.
        Create detailed descriptions, weights, and dimensions.
        """,
        
        "Inventory": """
        Generate inventory records for products in a warehouse.
        Include varying quantities, locations, and last check dates.
        Create realistic batch numbers and some expiry dates.
        """
    }
    
    # Define sample sizes for each schema
    sample_sizes = {
        "Supplier": 10,
        "Category": 12,
        "Product": 25,
        "Inventory": 35,
    }
    
    # Define custom generators for specific schema columns
    #
    # NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
    # using the schema descriptions. Custom generators give you precise control for fields where
    # you need specific distributions or formatting that might be challenging for the AI.
    #
    # Below we include only the most important custom generators as examples:
    #
    custom_generators = {
        "Supplier": {
            # Generate a specific distribution of active/inactive suppliers
            "active": lambda row, col: random.choices([True, False], weights=[0.8, 0.2])[0],
            # Generate values from a fixed set of options
            "payment_terms": lambda row, col: random.choice(["Net 30", "Net 60", "Net 15", "COD", "Prepaid"])
        },
        "Category": {
            # Control hierarchical relationships
            "parent_id": lambda row, col: random.choices([None, random.randint(1, 5)], weights=[0.7, 0.3])[0]
        },
        "Product": {
            # Just one example of numeric value control
            "price": lambda row, col: round(random.uniform(5.99, 499.99), 2)
        },
        "Inventory": {
            # Example of a date field with specific distribution
            "last_checked": lambda row, col: (datetime.datetime.now() - 
                datetime.timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d")
        }
    }
    
    # Generate data for all schemas with automatic dependency resolution
    # Specifying 'json' as the output_format
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes,
        custom_generators=custom_generators,
        output_dir=output_dir,
        output_format='json'  # This is the key parameter to save in JSON format
    )
    
    # Print some statistics about the generated data
    print("\n📊 Sample data statistics:\n")
    
    # Supplier statistics
    supplier_df = results["Supplier"]
    active_suppliers = supplier_df[supplier_df["active"] == True].shape[0]
    print(f"Supplier status distribution:")
    print(f"  - Active suppliers: {active_suppliers} ({active_suppliers/supplier_df.shape[0]*100:.1f}%)")
    print(f"  - Inactive suppliers: {supplier_df.shape[0] - active_suppliers} ({(supplier_df.shape[0] - active_suppliers)/supplier_df.shape[0]*100:.1f}%)")
    
    # Product statistics
    product_df = results["Product"]
    print(f"\nProduct price statistics:")
    print(f"  - Price range: ${product_df['price'].min():.2f} to ${product_df['price'].max():.2f}")
    print(f"  - Average price: ${product_df['price'].mean():.2f}")
    
    # Inventory statistics
    inventory_df = results["Inventory"]
    print(f"\nInventory quantity statistics:")
    print(f"  - Quantity range: {inventory_df['quantity'].min()} to {inventory_df['quantity'].max()} units")
    print(f"  - Average quantity: {inventory_df['quantity'].mean():.1f} units")
    print(f"  - Total inventory: {inventory_df['quantity'].sum()} units")
    
    # Verify foreign key relationships
    print("\n🔍 Verifying referential integrity:")
    
    # Check Category parent_id references
    valid_category_ids = set(results["Category"]["id"].tolist())
    invalid_parent_ids = [pid for pid in results["Category"]["parent_id"].dropna() if pid not in valid_category_ids]
    if invalid_parent_ids:
        print(f"  ❌ Invalid Category.parent_id references detected")
    else:
        print(f"  ✅ All Category.parent_id values reference valid Categories")
    
    # Check Product category_id references
    valid_product_category = all(cid in valid_category_ids for cid in results["Product"]["category_id"])
    if valid_product_category:
        print(f"  ✅ All Product.category_id values reference valid Categories")
    else:
        print(f"  ❌ Invalid Product.category_id references detected")
    
    # Check Product supplier_id references
    valid_supplier_ids = set(results["Supplier"]["id"].tolist())
    valid_product_supplier = all(sid in valid_supplier_ids for sid in results["Product"]["supplier_id"])
    if valid_product_supplier:
        print(f"  ✅ All Product.supplier_id values reference valid Suppliers")
    else:
        print(f"  ❌ Invalid Product.supplier_id references detected")
    
    # Check Inventory product_id references
    valid_product_ids = set(results["Product"]["id"].tolist())
    valid_inventory_product = all(pid in valid_product_ids for pid in results["Inventory"]["product_id"])
    if valid_inventory_product:
        print(f"  ✅ All Inventory.product_id values reference valid Products")
    else:
        print(f"  ❌ Invalid Inventory.product_id references detected")


if __name__ == "__main__":
    main()
