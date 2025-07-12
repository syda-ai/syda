# Custom Generators

Custom generators in SYDA provide a powerful way to control and customize the data generation process. They allow you to define specific logic for generating or transforming field values beyond what the LLM can do alone.

## When to Use Custom Generators

Custom generators are particularly useful when you need to:

1. Use existing data to generate values
2. Ensure consistency across related records for complex relationships
3. Compute values based on other fields
4. Implement complex business rules
5. Access data from related tables
6. Post-process LLM-generated content

## Creating Custom Generators

A custom generator is simply a Python function that follows this signature:

```python
def my_custom_generator(row, col_name=None, parent_dfs=None):
    """
    Generate or transform a value for a specific field.
    
    Args:
        row: The current row being processed (as a pandas Series or dict-like object)
        col_name: The name of the column being generated
        parent_dfs: Dictionary of previously generated dataframes (schema name as key)
        
    Returns:
        The value to use for this field
    """
    # Your custom logic here
    return generated_value
```

### Simple Example: Calculating Tax Amount

Here's a simple example that calculates tax amount based on subtotal and tax rate:

```python
def calculate_tax(row, col_name=None):
    """Calculate tax amount based on subtotal and tax rate."""
    return round(row["subtotal"] * row["tax_rate"] / 100, 2)

def calculate_total(row, col_name=None):
    """Calculate total from subtotal, tax, and discount."""
    return round(row["subtotal"] + row["tax_amount"] - row["discount_amount"], 2)
```

### Complex Example: Generating Receipt Items

Here's a more complex example from the retail example project that generates receipt items by accessing previously generated data from other tables:

```python
def generate_receipt_items(row, col_name=None, parent_dfs=None):
    """Custom generator for the items field in receipts.
    
    Uses the Product and Transaction data from parent_dfs to populate receipt items.
    
    Args:
        row: The current row being processed
        col_name: The name of the column being generated
        parent_dfs: Dictionary of previously generated dataframes (schema name as key)
    """
    items = []
    subtotal = 0.0
    
    try:
        # Get customer ID
        customer_id = row.get('customer_id', None)
        
        # Use the parent_dfs parameter which contains the generated data
        if parent_dfs and 'Product' in parent_dfs and 'Transaction' in parent_dfs:
            # Get Product and Transaction data from parent_dfs
            products_df = parent_dfs['Product']
            transactions_df = parent_dfs['Transaction']
            
            # Filter transactions for this customer if possible
            if customer_id and 'customer_id' in transactions_df.columns:
                customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
                if len(customer_transactions) == 0:
                    # If no matching transactions, just take random ones
                    customer_transactions = transactions_df.sample(min(5, len(transactions_df)))
            else:
                # If no customer ID or column, just take random transactions
                customer_transactions = transactions_df.sample(min(5, len(transactions_df)))
            
            # Process each transaction to create a receipt item
            for _, tx in customer_transactions.iterrows():
                # Get product information
                if 'product_id' in tx and tx['product_id'] is not None:
                    product_matches = products_df[products_df['id'] == tx['product_id']]
                    if len(product_matches) > 0:
                        product = product_matches.iloc[0]
                        
                        # Extract product details
                        product_name = product['name']
                        sku = product['sku']
                        unit_price = float(product['price'])
                        quantity = int(tx['quantity'])
                        
                        # Calculate item total
                        item_total = round(quantity * unit_price, 2)
                        
                        # Add to items list
                        items.append({
                            "product_name": product_name,
                            "sku": sku,
                            "quantity": quantity,
                            "unit_price": unit_price,
                            "item_total": item_total
                        })
                        
                        # Update subtotal
                        subtotal += item_total
        
        # Update the row's subtotal
        row["subtotal"] = round(subtotal, 2)
        
    except Exception as e:
        # Log any errors
        print(f"Error generating receipt items: {str(e)}")
        
    return items
```


## Registering Custom Generators

Custom generators are registered by passing them to the `generate_for_schemas` or `generate_for_sqlalchemy_models` method via the `custom_generators` parameter:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Define custom generators for field calculations
def calculate_tax(row, col_name=None):
    """Calculate tax amount based on subtotal and tax rate."""
    return round(row["subtotal"] * row["tax_rate"] / 100, 2)

def calculate_total(row, col_name=None):
    """Calculate total from subtotal, tax, and discount."""
    return round(row["subtotal"] + row["tax_amount"] - row["discount_amount"], 2)

def generate_receipt_items(row, col_name=None, parent_dfs=None):
    """Generate items for a receipt based on products and transactions."""
    # Implementation shown in earlier example
    return items

# Register custom generators - schema name as key, then field name as sub-key
custom_generators = {
    "Receipt": {
        "items": generate_receipt_items,
        "tax_amount": calculate_tax,
        "total": calculate_total
    }
}

# Initialize the generator
config = ModelConfig(
    provider="anthropic", 
    model="claude-3-haiku-20240307",
    max_tokens=8192
)
generator = SyntheticDataGenerator(model_config=config)

# Use custom generators during data generation
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes=sample_sizes,
    prompts=prompts,
    custom_generators=custom_generators,
    output_dir=OUTPUT_DIR
)
```



## Best Practices

1. **Keep Generators Simple**: Each generator should have a single responsibility
2. **Handle Missing Data**: Always check if required fields exist before using them
3. **Use Type Checking**: Verify data types before performing operations on values
4. **Add Error Handling**: Catch exceptions to prevent generator failures
5. **Document Your Generators**: Include clear docstrings that explain functionality
6. **Test in Isolation**: Test generators independently with sample data
7. **Avoid Side Effects**: Unless needed (like the receipt example), generators shouldn't modify unrelated state


## Examples

To see custom generator in action, explore  [SQLAlchemy Example](../examples/structured_and_unstructured_mixed/sqlalchemy_models.md) and [Yaml Example](../examples/structured_and_unstructured_mixed/yaml_schemas.md) 
