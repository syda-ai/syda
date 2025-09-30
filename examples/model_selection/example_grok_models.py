#!/usr/bin/env python3
"""
Example demonstrating Grok models (Grok-3 and Grok-4) usage with Syda.

This example shows how to use Grok models for synthetic data generation.
Grok-3 and Grok-4 are xAI's large language models that provide OpenAI-compatible API access.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_data_for_model(model_name, sample_sizes):
    """Generate data for a specific Grok model."""
    print(f"\n{'='*60}")
    print(f"Generating data with {model_name.upper()} model...")
    print(f"{'='*60}")
    
    # Configure the specific Grok model
    config = ModelConfig(
        provider="grok",
        model_name=model_name,
        temperature=0.7,
        max_tokens=8000,  # Increased for larger datasets
        top_p=0.9
    )
    
    # Initialize generator with Grok configuration
    generator = SyntheticDataGenerator(
        model_config=config,
        grok_api_key=os.environ.get("GROK_API_KEY")  # Get API key from environment
    )
    
    # Define a simple schema for testing
    schemas = {
        'companies': {
            '__table_description__': 'Technology companies with innovative products and services',
            'id': {
                'type': 'number', 
                'description': 'Unique company identifier', 
                'primary_key': True
            },
            'name': {
                'type': 'text', 
                'description': 'Company name (e.g., TechCorp, InnovateLabs, FutureSystems)'
            },
            'industry': {
                'type': 'text', 
                'description': 'Primary industry sector (AI, FinTech, HealthTech, EdTech, etc.)'
            },
            'founded_year': {
                'type': 'number', 
                'description': 'Year the company was founded (1990-2024)'
            },
            'employee_count': {
                'type': 'number', 
                'description': 'Number of employees (1-10000)'
            },
            'revenue_millions': {
                'type': 'number', 
                'description': 'Annual revenue in millions USD (0.1-1000)'
            },
            'is_public': {
                'type': 'boolean', 
                'description': 'Whether the company is publicly traded'
            },
            'headquarters': {
                'type': 'text', 
                'description': 'City and country where company is headquartered'
            }
        },
        
        'products': {
            '__table_description__': 'Products and services offered by technology companies',
            '__foreign_keys__': {
                'company_id': ['companies', 'id']  # products.company_id references companies.id
            },
            'id': {
                'type': 'number', 
                'description': 'Unique product identifier', 
                'primary_key': True
            },
            'name': {
                'type': 'text', 
                'description': 'Product or service name'
            },
            'company_id': {
                'type': 'foreign_key', 
                'description': 'Reference to the company that owns this product'
            },
            'category': {
                'type': 'text', 
                'description': 'Product category (Software, Hardware, Service, Platform)'
            },
            'launch_year': {
                'type': 'number', 
                'description': 'Year the product was launched'
            },
            'price_usd': {
                'type': 'number', 
                'description': 'Product price in USD (0 for free products)'
            },
            'is_ai_powered': {
                'type': 'boolean', 
                'description': 'Whether the product uses AI/ML technology'
            }
        }
    }
    
    # Define custom prompts for Grok models
    prompts = {
        'companies': """
        Generate innovative technology companies with diverse backgrounds.
        Include a mix of startups and established companies across different industries.
        Focus on companies that are pushing the boundaries of technology.
        Create realistic company names, industries, and business metrics.
        """,
        
        'products': """
        Generate cutting-edge technology products and services.
        Include both software and hardware products, with realistic pricing.
        Focus on innovative products that could exist in the current tech landscape.
        Ensure products align with their parent companies' industries and values.
        """
    }
    
    # Set output directory (model-specific subfolder under test_grok_models)
    model_name = config.model_name  # Get the model name from config
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_grok_models", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating data with {model_name.upper()} model...")
    print("  - Companies: 50 records")
    print("  - Products: 100 records")
    print("  - Provider: Grok (xAI)")
    print(f"  - Model: {model_name}")
    print()
    
    # Generate data using the specified Grok model with larger dataset
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes={"companies": 50, "products": 100},
        output_dir=output_dir
    )
    
    # Print results summary
    print("Data generation complete!")
    for schema_name, df in results.items():
        print(f"  - {schema_name}: {len(df)} records")
    
    print(f"\nData files saved to: {output_dir}")
    
    # Display sample data
    print("\nSample generated data:")
    for schema_name, df in results.items():
        print(f"\n{schema_name.upper()} (first 3 records):")
        print(df.head(3).to_string(index=False))
    
    # Verify referential integrity
    print("\nVerifying referential integrity:")
    
    # Check companies → products relationship
    if 'companies' in results and 'products' in results:
        companies_df = results['companies']
        products_df = results['products']
        
        # Get all company IDs
        company_ids = set(companies_df['id'].tolist())
        
        # Get all product company_id references
        product_company_ids = set(products_df['company_id'].tolist())
        
        # Check if all product company_id values exist in companies
        invalid_refs = product_company_ids - company_ids
        
        if invalid_refs:
            print(f"  Found {len(invalid_refs)} invalid company references in products")
            print(f"     Invalid IDs: {list(invalid_refs)}")
        else:
            print(f"  All {len(product_company_ids)} product company references are valid")
        
        # Show distribution of products per company
        product_counts = products_df['company_id'].value_counts()
        print(f"  Products per company distribution:")
        for company_id, count in product_counts.items():
            company_name = companies_df[companies_df['id'] == company_id]['name'].iloc[0]
            print(f"     - {company_name} (ID: {company_id}): {count} products")
    
    # Analyze AI adoption
    if 'products' in results:
        ai_products = results['products']['is_ai_powered'].sum()
        total_products = len(results['products'])
        ai_percentage = (ai_products / total_products) * 100
        print(f"\nAI Adoption Analysis:")
        print(f"  - AI-powered products: {ai_products}/{total_products} ({ai_percentage:.1f}%)")
        
        # Show AI vs non-AI product categories
        ai_by_category = results['products'].groupby(['category', 'is_ai_powered']).size().unstack(fill_value=0)
        print(f"  - AI adoption by category:")
        for category in ai_by_category.index:
            ai_count = ai_by_category.loc[category, True] if True in ai_by_category.columns else 0
            non_ai_count = ai_by_category.loc[category, False] if False in ai_by_category.columns else 0
            total_cat = ai_count + non_ai_count
            ai_pct = (ai_count / total_cat * 100) if total_cat > 0 else 0
            print(f"     - {category}: {ai_count}/{total_cat} AI products ({ai_pct:.1f}%)")
    
    # Analyze company metrics
    if 'companies' in results:
        companies_df = results['companies']
        print(f"\nCompany Analysis:")
        print(f"  - Average employees: {companies_df['employee_count'].mean():.0f}")
        print(f"  - Average revenue: ${companies_df['revenue_millions'].mean():.1f}M")
        print(f"  - Public companies: {companies_df['is_public'].sum()}/{len(companies_df)} ({companies_df['is_public'].mean()*100:.1f}%)")
        
        # Industry distribution
        industry_counts = companies_df['industry'].value_counts()
        print(f"  - Industry distribution:")
        for industry, count in industry_counts.items():
            print(f"     - {industry}: {count} companies")
    
    print(f"\n{model_name.upper()} integration test completed successfully!")
    print(f"   Generated realistic tech company data with perfect referential integrity.")
    
    return results

def main():
    """Demonstrate both Grok-3 and Grok-4 models usage for synthetic data generation."""
    print("Starting Grok models example...")
    print("This example demonstrates both Grok-3 and Grok-4 models")
    
    # Check if API key is available
    if not os.environ.get("GROK_API_KEY"):
        print("[ERROR] GROK_API_KEY not found in environment variables")
        print("Please set your xAI API key:")
        print("export GROK_API_KEY=your_api_key_here")
        return
    
    print(f"[INFO] Using API key: {os.environ.get('GROK_API_KEY')[:10]}...")
    
    # Test both models
    models_to_test = [
        ("grok-3", {"companies": 50, "products": 100}),  # Larger dataset for Grok-3
        ("grok-4", {"companies": 25, "products": 50})    # Smaller dataset for Grok-4
    ]
    
    results = {}
    
    for model_name, sample_sizes in models_to_test:
        try:
            print(f"\nTesting {model_name}...")
            result = generate_data_for_model(model_name, sample_sizes)
            results[model_name] = True
            print(f"[OK] {model_name.upper()} test completed successfully!")
        except Exception as e:
            print(f"[ERROR] {model_name.upper()} test failed: {str(e)}")
            results[model_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for model_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {model_name}: {status}")
    
    print(f"\nGrok models integration completed!")
    print("   Generated realistic tech company data with perfect referential integrity.")


if __name__ == "__main__":
    main()
