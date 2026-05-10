"""
Syda 30-Second Quick Start Example
Demonstrates AI-powered synthetic data generation with perfect referential integrity
"""

from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("🚀 Starting Syda Quick Start...")

# Configure AI model
generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        #provider="anthropic",
        #model_name="claude-3-5-haiku-20241022"
        #provider="openai",
        #model_name="gpt-4o"
        provider="deepseek",
        model_name="deepseek-chat"  # <-- use a valid DeepSeek model name
    )
)

# Define schemas with rich descriptions for better AI understanding
schemas = {
    'categories': {
        '__table_description__': 'Product categories for organizing items in the e-commerce catalog',
        'id': {
            'type': 'number',
            'description': 'Unique identifier for the category',
            'primary_key': True
        },
        'name': {
            'type': 'text',
            'description': 'Category name (Electronics, Home Decor, Sports, etc.)'
        },
        'description': {
            'type': 'text',
            'description': 'Detailed description of what products belong in this category'
        }
    },

    'products': {
        '__table_description__': 'Individual products available for purchase with pricing and category assignment',
        '__foreign_keys__': {
            'category_id': ['categories', 'id']  # products.category_id references categories.id
        },
        'id': {
            'type': 'number',
            'description': 'Unique product identifier',
            'primary_key': True
        },
        'name': {
            'type': 'text',
            'description': 'Product name and title'
        },
        'category_id': {
            'type': 'foreign_key',
            'description': 'Reference to the category this product belongs to'
        },
        'price': {
            'type': 'number',
            'description': 'Product price in USD'
        }
    }
}

# Generate data with perfect referential integrity
print("📊 Generating categories and products...")
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={"categories": 5, "products": 20},
    output_dir="data"
)

print("✅ Generated realistic data with perfect foreign key relationships!")
print("📂 Check the 'data' folder for categories.csv and products.csv")
# Check data/ folder for categories.csv and products.csv