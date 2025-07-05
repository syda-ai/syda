#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retail Sector Example Using YAML Schemas

This example demonstrates using YAML schemas to generate both structured data
and PDF documents for a retail sector application. It includes:
- Product categories
- Products
- Customers
- Transactions
- Receipt templates
"""

import os
import pandas as pd
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Define output directory - simplified paths since we're in the retail folder
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Run the retail sector example."""
    print("\nRetail Sector Example Using YAML Schemas")
    print("=========================================")
    
    # Initialize the generator with Claude
    config = ModelConfig(provider="anthropic", model="claude-3-haiku-20240307")
    generator = SyntheticDataGenerator(model_config=config)
    
    # Define schema directory paths - simplified paths since we're in the retail folder
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
    
    # Define paths to schema files in a single dictionary
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
    
    # We just need some basic calculators for financial values
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
        print("\n===== CUSTOM GENERATOR CALLED: generate_receipt_items =====")        
        import pandas as pd
        
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
                        
                        # Extract product details - use only actual values, no fallbacks
                        product_name = product['name']
                        sku = product['sku']
                        unit_price = float(product['price'])
                        quantity = int(tx['quantity'])
                        
                        # Calculate item total
                        item_total = round(quantity * unit_price, 2)
                        
                        # Debug product info
                        print(f"  - Adding product: {product_name}, SKU: {sku}, Qty: {quantity}, Price: ${unit_price}, Total: ${item_total}")
                        
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
                    else:
                        print(f"  - No product found with ID {tx['product_id']}")
                
            print(f"  - Generated {len(items)} items with subtotal: ${subtotal:.2f}")
            
        except Exception as e:
            print(f"Error generating receipt items: {str(e)}")
            import traceback
            traceback.print_exc()
                
        # Update the row's subtotal
        row["subtotal"] = round(subtotal, 2)
        
        # Do not use fallback data, just log the issue
        if not items:
            print("  ⚠️ Warning: No items were generated. The receipt will have an empty items section.")
            row["subtotal"] = 0.0
            
        print("\n===== ITEMS GENERATED =====")
        return items
            
    # Custom generators dictionary - simpler now with foreign keys
    custom_generators = {
        "Receipt": {
            "items": generate_receipt_items,
            "tax_amount": calculate_tax,
            "total": calculate_total
        }
    }
    
    print("\n🔄 Generating data for retail sector...")
    print("  The system will automatically determine the right generation order")
    print("  and handle foreign key relationships and template processing\n")
    

    # Generate data for all schemas in a single step
    # The __depends_on__ mechanism in receipt.yml ensures the correct generation order
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=OUTPUT_DIR,
        custom_generators=custom_generators
    )
    

    # Print summary of generated files
    print("\nGeneration Complete!")
    print("=" * 40)  
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    
    # Check for PDF files in template directories
    pdf_count = 0
    template_dir = os.path.join(OUTPUT_DIR, "Receipt")
    if os.path.exists(template_dir):
        pdf_files = [f for f in os.listdir(template_dir) if f.endswith('.pdf')]
        pdf_count += len(pdf_files)
    
    print(f"Total: {pdf_count} PDF documents")
    print("\nDone!")

if __name__ == "__main__":
    main()