#!/usr/bin/env python
"""
Example of using the SyntheticDataGenerator to handle multiple related YAML schema files
with embedded foreign key relationships.

This example demonstrates:
1. Loading schemas exclusively from YAML files
2. Foreign key relationships defined within the schema files themselves
3. Automatic dependency resolution between schemas
4. Custom generators for specific fields
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
    """Demonstrate automatic generation of related data using YAML schema files with embedded foreign keys."""
    
    # Create a generator instance with appropriate max_tokens setting
    model_config = ModelConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        temperature=0.7,
        max_tokens=8192,  # Using higher max_tokens value for more complete responses
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Define output directory
    output_dir = "inventory_data"
    
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
    
    # Define sample sizes for each schema (optional)
    sample_sizes = {
        "Supplier": 10,      # Base entities
        "Category": 12,      # Categories for products
        "Product": 25,       # Products across categories from suppliers
        "Inventory": 35,     # Inventory records for products (some products have multiple records)
    }
    
    # Define custom generators for specific schema columns
    custom_generators = {
        "Supplier": {
            # Generate a mix of active and inactive suppliers
            "active": lambda row, col: random.choices([True, False], weights=[0.8, 0.2])[0],
            # Generate realistic payment terms
            "payment_terms": lambda row, col: random.choice(["Net 30", "Net 60", "Net 15", "COD", "Prepaid"])
        },
        "Category": {
            # 70% of categories are top-level (no parent)
            "parent_id": lambda row, col: None if random.random() < 0.7 else row["parent_id"]
        },
        "Product": {
            # Generate prices between $5 and $500 with appropriate precision
            "price": lambda row, col: round(random.uniform(5, 500), 2),
            # Generate weights between 0.1 and 50 kg
            "weight": lambda row, col: round(random.uniform(0.1, 50), 2),
            # Generate dimensions in the format "LxWxH cm"
            "dimensions": lambda row, col: f"{random.randint(1, 100)}x{random.randint(1, 100)}x{random.randint(1, 100)} cm",
            # Generate in_stock status with 80% in stock
            "in_stock": lambda row, col: random.choices([True, False], weights=[0.8, 0.2])[0],
            # Generate reorder levels between 5 and 50
            "reorder_level": lambda row, col: random.randint(5, 50)
        },
        "Inventory": {
            # Generate quantities between 0 and 200
            "quantity": lambda row, col: random.randint(0, 200),
            # Generate warehouse locations
            "warehouse_location": lambda row, col: f"{random.choice('ABCDE')}-{random.randint(1, 9)}-{random.randint(1, 20)}",
            # Generate last checked dates in the last 90 days
            "last_checked": lambda row, col: (datetime.datetime.now() - 
                datetime.timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d"),
            # 50% chance of having an expiry date in the next 1-2 years
            "expiry_date": lambda row, col: None if random.random() < 0.5 else 
                (datetime.datetime.now() + datetime.timedelta(days=random.randint(365, 730))).strftime("%Y-%m-%d"),
            # Generate batch numbers
            "batch_number": lambda row, col: f"B{random.randint(10000, 99999)}",
            # Generate purchase order IDs
            "purchase_order_id": lambda row, col: f"PO-{random.randint(1000, 9999)}"
        }
    }
    
    print("\n🔄 Generating related data for Inventory system...")
    print("  The system will automatically extract foreign key relationships from YAML schema files")
    print("  and determine the correct generation order\n")
    
    # Generate data using schemas with embedded foreign keys
    # Note: We don't need to explicitly provide the foreign_keys parameter
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
    print("\n📊 Sample data statistics:")
    
    print("\nSupplier status distribution:")
    active_counts = results["Supplier"]["active"].value_counts().to_dict()
    print(f"  - Active suppliers: {active_counts.get(True, 0)} ({active_counts.get(True, 0)/len(results['Supplier'])*100:.1f}%)")
    print(f"  - Inactive suppliers: {active_counts.get(False, 0)} ({active_counts.get(False, 0)/len(results['Supplier'])*100:.1f}%)")
    
    print("\nProduct price statistics:")
    prices = results["Product"]["price"].tolist()
    print(f"  - Price range: ${min(prices):.2f} to ${max(prices):.2f}")
    print(f"  - Average price: ${sum(prices)/len(prices):.2f}")
    
    print("\nInventory quantity statistics:")
    quantities = results["Inventory"]["quantity"].tolist()
    print(f"  - Quantity range: {min(quantities)} to {max(quantities)} units")
    print(f"  - Average quantity: {sum(quantities)/len(quantities):.1f} units")
    print(f"  - Total inventory: {sum(quantities)} units")
    
    print("\n🔍 Verifying referential integrity:")
    
    # Check Categories → Parent Categories
    child_categories = results["Category"][results["Category"]["parent_id"].notna()]
    if len(child_categories) > 0:
        parent_ids = set(child_categories["parent_id"].tolist())
        valid_category_ids = set(results["Category"]["id"].tolist())
        if parent_ids.issubset(valid_category_ids):
            print("  ✅ All Category.parent_id values reference valid Categories")
        else:
            print("  ❌ Invalid Category.parent_id references detected")
    
    # Check Products → Categories
    product_category_ids = set(results["Product"]["category_id"].tolist())
    valid_category_ids = set(results["Category"]["id"].tolist())
    if product_category_ids.issubset(valid_category_ids):
        print("  ✅ All Product.category_id values reference valid Categories")
    else:
        print("  ❌ Invalid Product.category_id references detected")
    
    # Check Products → Suppliers
    product_supplier_ids = set(results["Product"]["supplier_id"].tolist())
    valid_supplier_ids = set(results["Supplier"]["id"].tolist())
    if product_supplier_ids.issubset(valid_supplier_ids):
        print("  ✅ All Product.supplier_id values reference valid Suppliers")
    else:
        print("  ❌ Invalid Product.supplier_id references detected")
    
    # Check Inventory → Products
    inventory_product_ids = set(results["Inventory"]["product_id"].tolist())
    valid_product_ids = set(results["Product"]["id"].tolist())
    if inventory_product_ids.issubset(valid_product_ids):
        print("  ✅ All Inventory.product_id values reference valid Products")
    else:
        print("  ❌ Invalid Inventory.product_id references detected")


if __name__ == "__main__":
    main()
