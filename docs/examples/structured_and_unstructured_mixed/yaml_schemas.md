# YAML Schemas with Mixed Content

> Source code: [examples/structured_and_unstructured/retail_yml/example_retail_schemas.py](https://github.com/syda-ai/syda/blob/main/examples/structured_and_unstructured/retail_yml/test_retail_schemas.py)

This example demonstrates how to use YAML schemas to generate both structured data and unstructured document content for a retail application.

## Overview

SYDA can generate both structured tabular data and unstructured document content like PDFs in a single workflow. This approach is useful when you need to maintain consistency between your structured database records and free-form documents that reference them.

In this retail example, we generate:
- Structured data: Product categories, products, customers, and transactions
- Unstructured content: Receipt documents (PDFs) that include items from transactions

## Schema Definition

The retail example uses YAML schemas to define both structured data models and document templates. 

### Structured Data Schemas

Here's an example of the structured YAML schemas:

```yaml
# category.yml
__table_description__: Product categories in the retail system
id:
  type: number
  description: Unique identifier for the category
  primary_key: true
name:
  type: text
  description: Category name
  nullable: false
parent_id:
  type: number
  description: Reference to parent category (if any, for hierarchical categories)
  nullable: true
description:
  type: text
  description: Category description
```

```yaml
# product.yml
__table_description__: Products available for purchase in the retail system
id:
  type: number
  description: Unique identifier for the product
  primary_key: true
name:
  type: text
  description: Product name
  nullable: false
sku:
  type: text
  description: Stock keeping unit (unique product code)
  nullable: false
category_id:
  type: number
  description: Reference to product category
  nullable: false
  references: Category.id
price:
  type: number
  description: Product price in USD
  nullable: false
description:
  type: text
  description: Detailed product description
```

### Document Template Schema

For generating unstructured documents (like receipts), we define a special schema with template processing capabilities:

```yaml
# receipt.yml
__table_description__: Retail receipts with store details and itemized purchases
__template__: true
__template_source__: templates/receipt.html
__input_file_type__: html
__output_file_type__: pdf
__depends_on__:
  - Customer
  - Transaction
  - Product

id:
  type: number
  description: Unique identifier for the receipt
  primary_key: true

store_name:
  type: text
  description: Name of the retail store
  nullable: false

store_address:
  type: text
  description: Full address of the retail store
  nullable: false

transaction_date:
  type: date
  description: Date when the transaction occurred
  nullable: false

customer_id:
  type: number
  description: Reference to customer who made the purchase
  references: Customer.id
  nullable: true

cashier:
  type: text
  description: Name of the cashier who processed the transaction
  nullable: false

items:
  type: array
  description: List of items purchased in this transaction
  item_fields:
    product_name:
      type: text
      description: Name of the product purchased
    sku:
      type: text
      description: Product SKU/code
    quantity:
      type: number
      description: Quantity purchased
    unit_price:
      type: number
      description: Price per unit
    item_total:
      type: number
      description: Total price for this item (quantity * unit_price)

subtotal:
  type: number
  description: Sum of all item totals before tax and discounts
  nullable: false

tax_rate:
  type: number
  description: Tax rate applied as percentage (e.g. 7.5 for 7.5%)
  nullable: false

tax_amount:
  type: number
  description: Calculated tax amount
  nullable: false

discount_amount:
  type: number
  description: Any discounts applied to the transaction
  nullable: false
  default: 0.0

total:
  type: number
  description: Final total including tax and discounts
  nullable: false

payment_method:
  type: text
  description: Method of payment (Credit, Cash, etc)
  nullable: false

notes:
  type: text
  description: Any additional notes for the receipt
  nullable: true
```

## Foreign Key Handling

YAML schemas handle foreign keys through explicit references:

```yaml
category_id:
  type: number
  description: Reference to product category
  nullable: false
  references: Category.id
```

Additionally, for template generation, you can specify schema dependencies using the `__depends_on__` property:

```yaml
__depends_on__:
  - Customer
  - Transaction
  - Product
```

This ensures that all required data is generated before processing templates.

## Code Example

Here's how to use YAML schemas to generate mixed structured and unstructured data:

```python
import os
import pandas as pd
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Define output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Run the retail sector example."""
    
    # Initialize the generator with Claude
    config = ModelConfig(provider="anthropic", model="claude-3-haiku-20240307")
    generator = SyntheticDataGenerator(model_config=config)
    
    # Define schema directory paths
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
    
    # Define paths to schema files
    schemas = {
        "Category": os.path.join(schema_dir, "category.yml"),
        "Customer": os.path.join(schema_dir, "customer.yml"),
        "Product": os.path.join(schema_dir, "product.yml"),
        "Transaction": os.path.join(schema_dir, "transaction.yml"),
        "Receipt": os.path.join(schema_dir, "receipt.yml")
    }
    
    # Define custom prompts for each schema (optional)
    prompts = {
        "Category": "Generate retail product categories with hierarchical structure.",
        "Product": "Generate retail products with names, SKUs, prices, and descriptions. Ensure a good variety of prices and categories.",
        "Customer": "Generate customer records for a retail business with realistic names, addresses, and email patterns.",
        "Transaction": "Generate retail transactions with realistic purchase patterns, payment methods, and item counts.",
        "Receipt": "Generate data for retail receipts including store details, transaction information, and itemized purchases."
    }
    
    # Define sample sizes for each schema
    sample_sizes = {
        "Category": 5,
        "Product": 25,
        "Customer": 10,
        "Transaction": 20,
        "Receipt": 5
    }
    
    # Helper functions for calculated fields
    def calculate_tax(row, col_name=None):
        """Calculate tax amount based on subtotal and tax rate."""
        return round(row["subtotal"] * row["tax_rate"] / 100, 2)
    
    def calculate_total(row, col_name=None):
        """Calculate total from subtotal, tax, and discount."""
        return round(row["subtotal"] + row["tax_amount"] - row["discount_amount"], 2)
        
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
                # Get Product data from parent_dfs
                products_df = parent_dfs['Product']
                transactions_df = parent_dfs['Transaction']
                    
            else:
                print("  - Required parent dataframes not available")
                return []
            
            # Filter transactions for this customer if possible
            if customer_id and 'customer_id' in transactions_df.columns:
                customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
                if len(customer_transactions) == 0:
                    print(f"  - No transactions found for customer ID {customer_id}, using random transactions")
                    # If no matching transactions, just take random ones
                    customer_transactions = transactions_df.sample(min(5, len(transactions_df)))
                else:
                    print(f"  - Found {len(customer_transactions)} transactions for customer ID {customer_id}")
            else:
                print(f"  - No customer ID or customer_id column, using random transactions")
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
            
            # If no items, set subtotal to 0
            if not items:
                row["subtotal"] = 0.0
                
            return items
            
        except Exception as e:
            print(f"Error generating receipt items: {str(e)}")
            return []
            
    # Custom generators dictionary
    custom_generators = {
        "Receipt": {
            "items": generate_receipt_items,
            "tax_amount": calculate_tax,
            "total": calculate_total
        }
    }
    
    # Generate data for all schemas in a single step
    # The __depends_on__ mechanism ensures the correct generation order
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=OUTPUT_DIR,
        custom_generators=custom_generators
    )
```

## Key Features

1. **Mixed Content Generation**: Generate both structured data and documents in one workflow
2. **Template Processing**: Convert structured data into formatted documents (HTML → PDF)
3. **Cross-Reference Consistency**: Ensure generated documents reference valid structured data
4. **Custom Generators**: Define functions that calculate values or extract data from other schemas
5. **Dependency Management**: Define which schemas must be generated before others

## Document Template Files

The `__template__` property in the Receipt schema references an HTML template file:

```html
{% raw %}
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{store_name}} - Receipt</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.5;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        .receipt-details {
            margin-bottom: 20px;
        }
        .items-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .items-table th, .items-table td {
            border-bottom: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        .items-table th {
            background-color: #f5f5f5;
        }
        .amount-section {
            width: 50%;
            float: right;
            margin-top: 10px;
        }
        .amount-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .final-total {
            font-weight: bold;
            font-size: 1.2em;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 2px solid #333;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{store_name}}</h1>
        <p>{{store_address}}</p>
    </div>
    
    <div class="receipt-details">
        <p><strong>Receipt #:</strong> {{id}}</p>
        <p><strong>Date:</strong> {{transaction_date}}</p>
        <p><strong>Cashier:</strong> {{cashier}}</p>
        {% if customer_id %}
        <p><strong>Customer ID:</strong> {{customer_id}}</p>
        {% endif %}
    </div>
    
    <table class="items-table">
        <thead>
            <tr>
                <th>Item</th>
                <th>SKU</th>
                <th>Quantity</th>
                <th>Unit Price</th>
                <th>Total</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td>{{item.product_name}}</td>
                <td>{{item.sku}}</td>
                <td>{{item.quantity}}</td>
                <td>${{item.unit_price}}</td>
                <td>${{item.item_total}}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div class="amount-section">
        <div class="amount-row">
            <span>Subtotal:</span>
            <span>${{subtotal}}</span>
        </div>
        <div class="amount-row">
            <span>Tax ({{tax_rate}}%):</span>
            <span>${{tax_amount}}</span>
        </div>
        {% if discount_amount > 0 %}
        <div class="amount-row">
            <span>Discount:</span>
            <span>-${{discount_amount}}</span>
        </div>
        {% endif %}
        <div class="amount-row final-total">
            <span>Total:</span>
            <span>${{total}}</span>
        </div>
    </div>
    
    <div style="clear: both;"></div>
    
    <div class="payment-section">
        <p><strong>Payment Method:</strong> {{payment_method}}</p>
        {% if notes %}
        <p><strong>Notes:</strong> {{notes}}</p>
        {% endif %}
    </div>
    
    <div class="footer">
        <p>Thank you for shopping at {{store_name}}!</p>
        <p>Please keep this receipt for your records.</p>
    </div>
</body>
</html>
{% endraw %}
```

## Best Practices

1. **Define Dependencies**: Use `__depends_on__` to specify which schemas must be generated first

2. **Use Custom Generators**: Create custom generators for complex logic like populating document items

3. **Access Parent Data**: Use the `parent_dfs` parameter in custom generators to access previously generated data

4. **Calculate Derived Values**: Use simple functions for calculated fields like totals and taxes

5. **Design Clean Templates**: Create well-structured templates with appropriate styling and conditional sections

## Sample Outputs

You can view sample outputs generated using these YAML schemas and templates here:

> [Example Retail YAML Outputs](https://github.com/syda-ai/syda/tree/main/examples/structured_and_unstructured/retail_yml/output)
