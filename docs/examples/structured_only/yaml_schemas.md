# YAML Schema Examples

> Source code: [examples/structured_only/example_yaml_schemas.py](https://github.com/syda-ai/syda/blob/main/examples/structured_only/example_yaml_schemas.py)

This example demonstrates how to define and use YAML-based schemas for synthetic data generation with SYDA.

## Overview

YAML schemas provide a clean, readable way to define your data structures in external files. This approach separates your schema definitions from your code, making it easier to manage complex data models.

## Schema Definition

YAML schemas use a hierarchical structure where:
- Each YAML file defines one schema/table
- Field definitions include types and descriptions
- Special keys like `__table_description__` provide metadata
- Foreign keys can be defined using multiple methods

Here are examples of YAML schema files for an inventory management system:

### Supplier Schema (supplier.yml)

```yaml
__table_description__: Suppliers who provide products to the inventory system

id:
  type: number
  description: Unique identifier for the supplier
  constraints:
    primary_key: true

company_name:
  type: text
  description: Name of the supplier company
  constraints:
    unique: true
    max_length: 200

contact_name:
  type: text
  description: Name of the primary contact person at the supplier
  constraints:
    max_length: 100

email:
  type: email
  description: Email address for the supplier
  constraints:
    unique: true
    max_length: 150

phone:
  type: text
  description: Phone number for the supplier
  constraints:
    max_length: 30

address:
  type: text
  description: Physical address of the supplier
  constraints:
    max_length: 300

website:
  type: text
  description: Supplier's website URL
  constraints:
    max_length: 200

payment_terms:
  type: text
  description: Payment terms for this supplier (e.g., Net 30, Net 60)
```

### Category Schema (category.yml)

```yaml
__table_description__: Product categories in the inventory system

id:
  type: number
  description: Unique identifier for the category
  constraints:
    primary_key: true

name:
  type: text
  description: Name of the category
  constraints:
    unique: true
    max_length: 100

description:
  type: text
  description: Description of the category and what types of products it contains
  constraints:
    max_length: 1000

parent_id:
  type: number
  description: Reference to the parent category (for hierarchical categories), indicate 0 if it is a parent category
```

### Product Schema (product.yml)

```yaml
__table_description__: Products in the inventory management system

id:
  type: number
  description: Unique identifier for the product
  constraints:
    primary_key: true

name:
  type: text
  description: Name of the product
  constraints:
    unique: true
    max_length: 150

category_id:
  type: foreign_key
  description: Reference to the product category
  references:
    schema: Category
    field: id

sku:
  type: text
  description: Stock Keeping Unit - unique product identifier
  constraints:
    unique: true
    max_length: 50

price:
  type: number
  description: Current price of the product in USD
```

## Schema Directory Structure

For an inventory management system, you might structure your YAML schemas as follows:

```
schema_files/yaml/
├── supplier.yml
├── category.yml
├── product.yml
└── inventory.yml
```

## Foreign Key Handling

YAML schemas support three methods for defining foreign key relationships:

### 1. Using Field-Level References (Recommended for YAML)

```yaml
category_id:
  type: foreign_key
  description: Reference to the product category
  references:
    schema: Category
    field: id
```

### 2. Using the `__foreign_keys__` Special Section

```yaml
__foreign_keys__:
  category_id: [Category, id]  # product.category_id references category.id
```

## Code Example

Here's how to use YAML-based schemas with the SyntheticDataGenerator:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig
from dotenv import load_dotenv
import os
import random
import datetime

# Load environment variables from .env file
load_dotenv()

# Create a generator instance
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192
)
generator = SyntheticDataGenerator(model_config=model_config)

# Define paths to schema files
schema_dir = "schema_files/yaml"
schemas = {
    "Supplier": os.path.join(schema_dir, "supplier.yml"),
    "Category": os.path.join(schema_dir, "category.yml"),
    "Product": os.path.join(schema_dir, "product.yml"),
    "Inventory": os.path.join(schema_dir, "inventory.yml")
}

# Define custom prompts
prompts = {
    "Supplier": """Generate diverse suppliers for an inventory system.
        Include international suppliers with varied payment terms.
        Create a mix of large established vendors and smaller specialty suppliers.""",
        
    "Category": """Generate product categories for an inventory system.
        Include both parent categories and subcategories.
        Use realistic department store or e-commerce categories.""",
        
    "Product": """Generate diverse products across different categories.
        Include realistic prices, SKUs, and descriptions.
        Generate both popular and niche products.""",
        
    "Inventory": """Generate inventory records for products in a warehouse.
        Include varying quantities, locations, and last check dates.
        Create realistic batch numbers and some expiry dates."""
}

# Define sample sizes
sample_sizes = {
    "Supplier": 10,      # Base entities
    "Category": 12,      # Categories for products
    "Product": 25,       # Products across categories from suppliers
    "Inventory": 35,     # Inventory records for products (some products have multiple records)
}

# Define custom generators for specific schema fields
# NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
# using the schema descriptions. Custom generators give you precise control for fields where
# you need specific distributions or formatting.
def supplier_active_generator(row, col):
    return random.choices([True, False], weights=[0.8, 0.2])[0]

def supplier_payment_terms_generator(row, col):
    return random.choice(["Net 30", "Net 60", "Net 15", "COD", "Prepaid"])

def product_price_generator(row, col):
    return round(random.uniform(5.99, 499.99), 2)

def inventory_last_checked_generator(row, col):
    return (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d")

custom_generators = {
    "Supplier": {
        # Generate a specific distribution of active/inactive suppliers
        "active": supplier_active_generator,
        # Generate values from a fixed set of options
        "payment_terms": supplier_payment_terms_generator
    },
    "Product": {
        # Control numeric value distributions
        "price": product_price_generator
    },
    "Inventory": {
        # Control date field with specific distribution
        "last_checked": inventory_last_checked_generator
    }
}

# Generate data
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir="output/example_yaml_schemas/inventory_data",
    custom_generators=custom_generators
)
```

## Key Features

1. **External Files**: Maintain schema definitions separate from code
2. **Readability**: YAML's syntax is clean and human-readable
3. **Schema Evolution**: Easily update schema files without changing code
4. **Rich Constraints**: Define constraints like uniqueness, min/max values
5. **Automatic Generation Order**: SYDA handles dependency resolution

## Best Practices

1. **Organize by Domain**: Group related schemas in directories
2. **Include Descriptions**: Always add descriptions for fields and tables
3. **Version Control**: Track schema changes with your source control system
4. **Explicit Foreign Keys**: Clearly define relationships between tables

## Sample Outputs

The generator produces pandas DataFrames for each schema, which can be further processed or saved to various formats:

```python
# Print summary
for schema_name, df in results.items():
    print(f"{schema_name}: {len(df)} records")
    print(df.head(2))
```

You can view sample outputs generated using these YAML schemas here:

> [Example YAML Schema Outputs](https://github.com/syda-ai/syda/tree/main/examples/structured_only/output/example_yaml_schemas/inventory_data)
