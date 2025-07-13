# Dictionary-Based Schema Examples

> Source code: [examples/structured_only/example_dict_schemas.py](https://github.com/syda-ai/syda/blob/main/examples/structured_only/example_dict_schemas.py)

This example demonstrates how to define and use dictionary-based schemas directly in your Python code for synthetic data generation with SYDA.

## Overview

Dictionary-based schemas are the most straightforward way to define your data structure in SYDA. You define your schemas as Python dictionaries right in your code, without the need for external schema files.

## Schema Definition

Dictionary schemas are defined as nested Python dictionaries where:
- The top-level keys are table/schema names
- Each schema contains field definitions with types and descriptions
- Special keys like `__table_description__` and `__foreign_keys__` provide additional metadata

Here's an example of a dictionary-based schema for an e-commerce system:

```python
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
        'status': {'type': 'text', 'description': 'Current status of the order'},
        'total_amount': {'type': 'number', 'description': 'Total amount of the order in USD'}
    }
}
```

## Foreign Key Handling

Dictionary schemas support three methods for defining foreign key relationships:

### 1. Using the `__foreign_keys__` Special Section (Recommended)

```python
'Order': {
    '__foreign_keys__': {
        'customer_id': ['Customer', 'id']  # Order.customer_id references Customer.id
    },
    # field definitions...
}
```

### 2. Using Field-Level References

```python
'customer_id': {
    'type': 'foreign_key',
    'description': 'Reference to the customer who placed the order',
    'references': {
        'schema': 'Customer', 
        'field': 'id'
    }
}
```

## Code Example

Here's how to use dictionary-based schemas with the SyntheticDataGenerator:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig
import random

# Create a generator instance
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192
)
generator = SyntheticDataGenerator(model_config=model_config)

# Define output directory
output_dir = "output/example_dict_schemas/ecommerce"

# Define custom prompts (optional)
prompts = {
    "Customer": "Generate diverse customers for an e-commerce platform."
               "Include various loyalty tiers (Bronze, Silver, Gold, Platinum)"
               "and realistic signup dates within the last 3 years.",
    "Product": "Generate diverse products for an e-commerce store."
              "Include various categories (Electronics, Clothing, Home, Books, etc.)"
              "with realistic prices and descriptions.",
    "Order": "Generate realistic orders with appropriate dates and statuses"
            "(Pending, Processing, Shipped, Delivered, Cancelled)."
            "Total amounts should reflect typical e-commerce purchases."
}

# Define sample sizes
sample_sizes = {
    "Customer": 10,       # Base entities
    "Product": 15,        # Product catalog
    "Order": 25,          # ~2-3 orders per customer
    "OrderItem": 50,      # ~2 items per order
}

# Define custom generators for specific schema fields
# NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
# based on field names, types, and descriptions. Custom generators give you precise control 
# for fields where you need specific distributions or formatting.
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

# Generate data
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir=output_dir,
    custom_generators=custom_generators
)
```

## Key Features

1. **Inline Definition**: Define your schema directly in your Python code
2. **Rich Metadata**: Add descriptions for tables and fields
3. **Foreign Key Support**: Define relationships between tables
4. **Custom Generators**: Override AI generation for specific fields
5. **Automatic Generation Order**: SYDA handles generating parent tables before child tables

## Best Practices

1. **Use Descriptions**: Always include detailed descriptions for your schema and fields
2. **Explicit Foreign Keys**: Be explicit about foreign key relationships
3. **Custom Generators**: Use custom generators for fields that need specific formats or distributions

## Sample Outputs

The generator produces pandas DataFrames for each schema, which can be further processed or saved to various formats:

```python
# Print summary
for schema_name, df in results.items():
    print(f"{schema_name}: {len(df)} records")
    print(df.head(2))
```

You can view sample outputs generated using these dictionary schemas here:

> [Example Dictionary Schema Outputs](https://github.com/syda-ai/syda/tree/main/examples/structured_only/output/example_dict_schemas/ecommerce)
