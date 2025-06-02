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
from syda.generate import SyntheticDataGenerator
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
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_files/yaml")
    
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
    #
    # NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
    # using the schema descriptions. Custom generators give you precise control for fields where
    # you need specific distributions or formatting that might be challenging for the AI.
    #
    # Below we include only the most important custom generators as examples:
    #
    
    # Custom generator call counters for verification
    call_counters = {
        "Supplier.active": 0,
        "Supplier.payment_terms": 0,
        "Product.price": 0,
        "Inventory.last_checked": 0
    }
    
    # Define generators with tracking
    def supplier_active_generator(row, col):
        call_counters["Supplier.active"] += 1
        print(f"✓ Custom generator called: Supplier.active (Call #{call_counters['Supplier.active']})")
        return random.choices([True, False], weights=[0.8, 0.2])[0]
    
    def supplier_payment_terms_generator(row, col):
        call_counters["Supplier.payment_terms"] += 1
        print(f"✓ Custom generator called: Supplier.payment_terms (Call #{call_counters['Supplier.payment_terms']})")
        return random.choice(["Net 30", "Net 60", "Net 15", "COD", "Prepaid"])
    
    def product_price_generator(row, col):
        call_counters["Product.price"] += 1
        print(f"✓ Custom generator called: Product.price (Call #{call_counters['Product.price']})")
        return round(random.uniform(5.99, 499.99), 2)
    
    def inventory_last_checked_generator(row, col):
        call_counters["Inventory.last_checked"] += 1
        print(f"✓ Custom generator called: Inventory.last_checked (Call #{call_counters['Inventory.last_checked']})")
        return (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d")
    
    custom_generators = {
        "Supplier": {
            # Generate a specific distribution of active/inactive suppliers
            "active": supplier_active_generator,
            # Generate values from a fixed set of options
            "payment_terms": supplier_payment_terms_generator
        },
        "Category": {
            # No custom generators needed for Category
        },
        "Product": {
            # Just one example of numeric value control
            "price": product_price_generator
        },
        "Inventory": {
            # Example of a date field with specific distribution
            "last_checked": inventory_last_checked_generator
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
    
    # Verify custom generators were called
    print("\n🔍 Verifying custom generators:")
    for generator_name, count in call_counters.items():
        schema_name = generator_name.split('.')[0]
        field_name = generator_name.split('.')[1]
        expected_count = sample_sizes.get(schema_name, 0)
        
        if count == 0:
            print(f"  ❌ Custom generator for {generator_name} was NEVER called!")
        elif count != expected_count:
            print(f"  ⚠️ Custom generator for {generator_name} was called {count} times (expected {expected_count})")
        else:
            print(f"  ✅ Custom generator for {generator_name} was called correctly ({count} times)")
    
    # Verify data distributions for custom generators
    print("\n🔍 Verifying data distributions:")
    
    # Supplier.active - should be ~80% True, ~20% False
    if "Supplier" in results:
        active_true_count = results["Supplier"]["active"].sum()
        active_total = len(results["Supplier"])
        active_true_pct = (active_true_count / active_total) * 100
        print(f"  Supplier.active: {active_true_pct:.1f}% True (expected ~80%)")
        
        # Supplier.payment_terms - should have variety
        payment_terms_counts = results["Supplier"]["payment_terms"].value_counts()
        print(f"  Supplier.payment_terms distribution: {dict(payment_terms_counts)}")
    
    # Product.price - should be between 5.99 and 499.99
    if "Product" in results:
        price_min = results["Product"]["price"].min()
        price_max = results["Product"]["price"].max()
        print(f"  Product.price range: ${price_min:.2f} to ${price_max:.2f} (expected $5.99-$499.99)")
    
    # Inventory.last_checked - should be within last 90 days
    if "Inventory" in results:
        today = datetime.datetime.now().date()
        last_checked_dates = pd.to_datetime(results["Inventory"]["last_checked"]).dt.date
        oldest_date = min(last_checked_dates)
        newest_date = max(last_checked_dates)
        oldest_days_ago = (today - oldest_date).days
        newest_days_ago = (today - newest_date).days
        print(f"  Inventory.last_checked range: {newest_days_ago}-{oldest_days_ago} days ago (expected 1-90 days)")
    
    # Sample the generated data
    print("\n📊 Sample of generated data:")
    for schema_name, df in results.items():
        if not df.empty:
            print(f"\n{schema_name} (sample of first 2 records):")
            print(df.head(2).to_string())
    
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
    
    # Since parent_id is just a numeric field (not a foreign key), we only need to check
    # that child categories (those with parent_id > 0) reference valid category IDs
    child_categories = results["Category"][
        (results["Category"]["parent_id"].notna()) & 
        (results["Category"]["parent_id"] > 0)  # parent_id=0 indicates it's a parent category
    ]
    if len(child_categories) > 0:
        # For hierarchical relationships, just print the distribution
        parent_categories = results["Category"][results["Category"]["parent_id"] == 0]
        child_categories = results["Category"][results["Category"]["parent_id"] > 0]
        print(f"  ℹ️ Category hierarchy: {len(parent_categories)} parent categories, {len(child_categories)} child categories")
    else:
        print("  ℹ️ No hierarchical categories detected (all are parent categories)")

    
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
