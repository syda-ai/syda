# 🎯 Syda - AI-Powered Synthetic Data Generation

[![PyPI version](https://badge.fury.io/py/syda.svg?v=1)](https://badge.fury.io/py/syda)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-python.syda.ai-brightgreen.svg)](https://python.syda.ai)
[![GitHub stars](https://img.shields.io/github/stars/syda-ai/syda.svg)](https://github.com/syda-ai/syda/stargazers)

> **Generate high-quality synthetic data with AI while preserving referential integrity**

Syda seamlessly integrates with **Anthropic Claude**, **OpenAI GPT** and **Google Gemini** models to create realistic test data, maintain privacy compliance, and accelerate development workflows.

## 📚 Documentation

**📖 For detailed documentation, examples, and API reference, visit: [https://python.syda.ai/](https://python.syda.ai/)**

## ⚡ 30-Second Quick Start

```bash
pip install syda
```

Create `.env` file:
```bash
# .env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
# OR
GEMINI_API_KEY=your_gemini_api_key_here
```

```python
"""
Syda 30-Second Quick Start Example
Demonstrates AI-powered synthetic data generation with perfect referential integrity
"""

from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("🚀 Starting Syda 30-Second Quick Start...")

# Configure AI model
generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="anthropic", 
        model_name="claude-3-5-haiku-20241022"
    )
)

# Define schemas with rich descriptions for better AI understanding
schemas = {
    # Categories schema with table and column descriptions
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

    # Products schema with table and column descriptions and foreign keys
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
```


## 🚀 Why Developers Love Syda

| Feature | Benefit | Example |
|---------|---------|---------|
| 🤖 **Multi-AI Provider** | No vendor lock-in | Claude, GPT, Gemini models |
| 🔗 **Zero Orphaned Records** | Perfect referential integrity | `product.category_id` → `category.id` ✅ |
| 🏗️ **SQLAlchemy Native** | Use existing models directly | `Customer`, `Contact` classes → CSV data |
| 📊 **Multiple Schema Formats** | Flexible input options | SQLAlchemy, YAML, JSON, Dict |
| 📄 **Document Generation** | AI-powered PDFs linked to data | Product catalogs, receipts, contracts |
| 🔧 **Custom Generators** | Complex business logic | Tax calculations, pricing rules, arrays |
| 🛡️ **Privacy-First** | Protect real user data | GDPR/CCPA compliant testing |
| ⚡ **Developer Experience** | Just works | Type hints, great docs |


## 🛒 Retail Example

### 1. Define your schemas

<details>
<summary><strong>📋 Click to view schema files</strong> (category_schema.yml & product_schema.yml)</summary>

**category_schema.yml:**
```yaml
__table_name__: Category
__description__: Retail product categories

id:
  type: integer
  description: Unique category ID
  constraints:
    primary_key: true
    not_null: true
    min: 1
    max: 1000

name:
  type: string
  description: Category name
  constraints:
    not_null: true
    length: 50
    unique: true

parent_id:
  type: integer
  description: Parent category ID for hierarchical categories, if it is a parent category, this field should be 0
  constraints:
    min: 0
    max: 1000

description:
  type: text
  description: Detailed category description
  constraints:
    length: 500

active:
  type: boolean
  description: Whether the category is active
  constraints:
    not_null: true
```

**product_schema.yml:**
```yaml
__table_name__: Product
__description__: Retail products
__foreign_keys__:
  category_id: [Category, id]

id:
  type: integer
  description: Unique product ID
  constraints:
    primary_key: true
    not_null: true
    min: 1
    max: 10000

name:
  type: string
  description: Product name
  constraints:
    not_null: true
    length: 100
    unique: true

category_id:
  type: integer
  description: Category ID for the product
  constraints:
    not_null: true
    min: 1
    max: 1000

sku:
  type: string
  description: Stock Keeping Unit - unique product code
  constraints:
    not_null: true
    pattern: '^P[A-Z]{2}-\d{5}$'
    length: 10
    unique: true

price:
  type: float
  description: Product price in USD
  constraints:
    not_null: true
    min: 0.99
    max: 9999.99
    decimals: 2

stock_quantity:
  type: integer
  description: Current stock level
  constraints:
    not_null: true
    min: 0
    max: 10000

is_featured:
  type: boolean
  description: Whether the product is featured
  constraints:
    not_null: true
```

</details>



### 2. Generate structured data

<details>
<summary><strong>🐍 Click to view Python code</strong></summary>

```python
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure your AI model  
config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022"
)

# Create generator
generator = SyntheticDataGenerator(model_config=config)

# Define your schemas (structured data only)
schemas = {
    "categories": "category_schema.yml",
    "products": "product_schema.yml"
}

# Generate synthetic data with relationships intact
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={"categories": 5, "products": 20},
    output_dir="output",
    prompts = {
        "Category": "Generate retail product categories with hierarchical structure.",
        "Product": "Generate retail products with names, SKUs, prices, and descriptions. Ensure a good variety of prices and categories."
    }
)

# Perfect referential integrity guaranteed! 🎯
print("✅ Generated realistic data with perfect foreign key relationships!")
```

</details>

**Output:**
```bash
📂 output/
├── 📊 categories.csv    # 5 product categories with hierarchical structure
└── 📊 products.csv      # 20 products, all with valid category_id references
```

### 3. Want to generate documents too? Add document templates!

To generate **AI-powered documents** along with your structured data, simply add the product catalog schema and update your code:

<details>
<summary><strong>📄 Click to view document schema</strong> (product_catalog_schema.yml)</summary>

**product_catalog_schema.yml (Document Template):**
```yaml
__template__: true
__description__: Product catalog page template
__name__: ProductCatalog
__depends_on__: [Product, Category]
__foreign_keys__:
  product_name: [Product, name]
  category_name: [Category, name]
  product_price: [Product, price]
  product_sku: [Product, sku]
__template_source__: templates/product_catalog.html
__input_file_type__: html
__output_file_type__: pdf

# Product information (linked to Product table)
product_name:
  type: string
  length: 100
  description: Name of the featured product

category_name:
  type: string
  length: 50
  description: Category this product belongs to

product_sku:
  type: string
  length: 10
  description: Product SKU code

product_price:
  type: float
  decimals: 2
  description: Product price in USD

# Marketing content (AI-generated)
product_description:
  type: text
  length: 500
  description: Detailed marketing description of the product

key_features:
  type: text
  length: 300
  description: Bullet points of key product features

marketing_tagline:
  type: string
  length: 100
  description: Catchy marketing tagline for the product

availability_status:
  type: string
  enum: ["In Stock", "Limited Stock", "Out of Stock", "Pre-Order"]
  description: Current availability status
```

</details>

<details>
<summary><strong>🎨 Click to view HTML template</strong> (templates/product_catalog.html)</summary>

**Create the Jinja HTML template** (`templates/product_catalog.html`):
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ product_name }} - Product Catalog</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .catalog-page {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .product-header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }
        .product-name {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .category-sku {
            font-size: 16px;
            color: #7f8c8d;
            margin-bottom: 15px;
        }
        .price {
            font-size: 32px;
            color: #e74c3c;
            font-weight: bold;
        }
        .tagline {
            font-style: italic;
            font-size: 18px;
            color: #34495e;
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 8px;
        }
        .description {
            font-size: 16px;
            line-height: 1.6;
            margin: 25px 0;
            text-align: justify;
        }
        .features {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
        }
        .features h3 {
            color: #2c3e50;
            margin-top: 0;
        }
        .availability {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            margin-top: 30px;
        }
        .in-stock { background: #d4edda; color: #155724; }
        .limited-stock { background: #fff3cd; color: #856404; }
        .out-of-stock { background: #f8d7da; color: #721c24; }
        .pre-order { background: #d1ecf1; color: #0c5460; }
    </style>
</head>
<body>
    <div class="catalog-page">
        <div class="product-header">
            <div class="product-name">{{ product_name }}</div>
            <div class="category-sku">{{ category_name }} Category | SKU: {{ product_sku }}</div>
            <div class="price">${{ "%.2f"|format(product_price) }}</div>
        </div>
        
        <div class="tagline">"{{ marketing_tagline }}"</div>
        
        <div class="description">
            {{ product_description }}
        </div>
        
        <div class="features">
            <h3>KEY FEATURES:</h3>
            {{ key_features }}
        </div>
        
        <div class="availability {{ availability_status.lower().replace(' ', '-') }}">
            Availability: {{ availability_status }}
        </div>
    </div>
</body>
</html>
```

</details>

<details>
<summary><strong>🐍 Click to view updated Python code</strong> (with document generation)</summary>

```python
# Same setup as before...
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

load_dotenv()
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)

# Define your schemas (structured data)
schemas = {
    "categories": "category_schema.yml",
    "products": "product_schema.yml",
    # 🆕 Add document templates
    "product_catalogs": "product_catalog_schema.yml"
}


# Generate both structured data AND documents
results = generator.generate_for_schemas(
    schemas=schemas,
    templates=templates,  # 🆕 Add this line
    sample_sizes={
      "categories": 5,
      "products": 20,
      "product_catalogs": 10 # 🆕 Add this line
    },
    output_dir="output",
    prompts = {
        "Category": "Generate retail product categories with hierarchical structure.",
        "Product": "Generate retail products with names, SKUs, prices, and descriptions. Ensure a good variety of prices and categories.",
        "ProductCatalog": "Generate compelling product catalog pages with marketing descriptions, key features, and sales copy."  # 🆕 Add this line
    }
)

print("✅ Generated structured data + AI-powered product catalogs!")
```

</details>

**Enhanced Output:**
```bash
📂 output/
├── 📊 categories.csv           # 5 product categories with hierarchical structure
├── 📊 products.csv             # 20 products, all with valid category_id references  
└── 📄 product_catalogs/        # 🆕 AI-generated marketing documents
    ├── catalog_1.pdf           # Product names match products.csv
    ├── catalog_2.pdf           # Prices match products.csv
    ├── catalog_3.pdf           # Perfect data consistency!
    ├── ...
    └── catalog_10.pdf
```



## 📊 See It In Action

### **Realistic Retail Data + AI-Generated Product Catalogs**

**Categories Table:**
```csv
id,name,parent_id,description,active
1,Electronics,0,Electronic devices and accessories,true
2,Smartphones,1,Mobile phones and accessories,true
3,Laptops,1,Portable computers and accessories,true
4,Clothing,0,Apparel and fashion items,true
5,Men's Clothing,4,Men's apparel and accessories,true
```

**Products Table (with matching category_id):**
```csv
id,name,category_id,sku,price,stock_quantity,is_featured
1,iPhone 15 Pro,2,PSM-12345,999.99,50,true
2,MacBook Air M3,3,PLA-67890,1299.99,25,true
3,Samsung Galaxy S24,2,PSA-11111,899.99,75,false
4,Dell XPS 13,3,PDE-22222,1099.99,30,false
5,Men's Cotton T-Shirt,5,PMC-33333,24.99,200,false
```

**Generated Product Catalog PDF Content:**
```
IPHONE 15 PRO
Smartphones Category | SKU: PSM-12345

$999.99

Revolutionary Performance, Unmatched Design

Experience the future of mobile technology with the iPhone 15 Pro. 
Featuring the powerful A17 Pro chip, this device delivers unprecedented 
performance for both work and play. The titanium design combines 
durability with elegance, while the advanced camera system captures 
professional-quality photos and videos.

KEY FEATURES:
• A17 Pro chip with 6-core GPU
• Pro camera system with 3x optical zoom  
• Titanium design with Action Button
• USB-C connectivity
• All-day battery life

"Innovation that fits in your pocket"

Availability: In Stock
```

> 🎯 **Perfect Integration**: The PDF catalog contains **actual product names, SKUs, and prices** from the CSV data, plus **AI-generated marketing content** - zero inconsistencies!


### 4. Need custom business logic? Add custom generators!

For advanced scenarios requiring **custom calculations** or **complex business rules**, you can add custom generator functions:

<details>
<summary><strong>🔧 Click to view custom generators example</strong></summary>

```python
# Define custom generator functions
def calculate_tax(row, parent_dfs=None, **kwargs):
    """Calculate tax amount based on subtotal and tax rate"""
    subtotal = row.get('subtotal', 0)
    tax_rate = row.get('tax_rate', 8.5)  # Default 8.5%
    return round(subtotal * (tax_rate / 100), 2)

def calculate_total(row, parent_dfs=None, **kwargs):
    """Calculate final total: subtotal + tax - discount"""
    subtotal = row.get('subtotal', 0)
    tax_amount = row.get('tax_amount', 0)
    discount = row.get('discount_amount', 0)
    return round(subtotal + tax_amount - discount, 2)

def generate_receipt_items(row, parent_dfs=None, **kwargs):
    """Generate receipt items based on actual transactions"""
    items = []
    
    if parent_dfs and 'Product' in parent_dfs and 'Transaction' in parent_dfs:
        products_df = parent_dfs['Product']
        transactions_df = parent_dfs['Transaction']
        
        # Get customer's transactions
        customer_id = row.get('customer_id')
        customer_transactions = transactions_df[
            transactions_df['customer_id'] == customer_id
        ]
        
        # Build receipt items from actual transaction data
        for _, tx in customer_transactions.iterrows():
            product = products_df[products_df['id'] == tx['product_id']].iloc[0]
            
            items.append({
                "product_name": product['name'],
                "sku": product['sku'],
                "quantity": int(tx['quantity']),
                "unit_price": float(product['price']),
                "item_total": round(tx['quantity'] * product['price'], 2)
            })
    
    return items

# Add custom generators to your generation
custom_generators = {
    "ProductCatalog": {
        "tax_amount": calculate_tax,
        "total": calculate_total,
        "items": generate_receipt_items
    }
}

# Generate with custom business logic
results = generator.generate_for_schemas(
    schemas=schemas,
    templates=templates,
    sample_sizes={"categories": 5, "products": 20, "product_catalogs": 10},
    output_dir="output",
    custom_generators=custom_generators,  # 🆕 Add this line
    prompts={
        "Category": "Generate retail product categories with hierarchical structure.",
        "Product": "Generate retail products with names, SKUs, prices, and descriptions.",
        "ProductCatalog": "Generate compelling product catalog pages with marketing copy."
    }
)

print("✅ Generated data with custom business logic!")
```

</details>

> 🎯 **Custom generators let you:**
> - **Calculate fields** based on other data (taxes, totals, discounts)
> - **Access related data** from other tables via `parent_dfs`
> - **Implement complex business rules** (pricing logic, inventory rules)
> - **Generate structured data** (arrays, nested objects, JSON)

## 🏗️ Works with Your Existing SQLAlchemy Models

Already using **SQLAlchemy**? Syda works directly with your existing models - no schema conversion needed!

<details>
<summary><strong>🏗️ Click to view SQLAlchemy example</strong></summary>

```python
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# Your existing SQLAlchemy models
class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, comment='Customer organization name')
    industry = Column(String(50), comment='Industry sector')
    annual_revenue = Column(Float, comment='Annual revenue in USD')
    status = Column(String(20), comment='Active, Inactive, or Prospect')
    
    # Relationships work perfectly
    contacts = relationship("Contact", back_populates="customer")

class Contact(Base):
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    position = Column(String(100), comment='Job title')
    is_primary = Column(Boolean, comment='Primary contact for customer')
    
    customer = relationship("Customer", back_populates="contacts")

# Generate data directly from your models
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)

results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Customer, Contact],
    sample_sizes={"Customer": 10, "Contact": 25},
    output_dir="crm_data"
)

print("✅ Generated CRM data with perfect foreign key relationships!")
```

</details>

**Output:**
```bash
📂 crm_data/
├── 📊 customers.csv     # 10 companies with realistic industry data
└── 📊 contacts.csv      # 25 contacts, all with valid customer_id references
```

> 🎯 **Zero Configuration**: Your SQLAlchemy `comments` become AI generation hints, `ForeignKey` relationships are automatically maintained, and `nullable=False` constraints are respected!


## 🤝 Contributing

We would **love your contributions**! Syda is an open-source project that thrives on community involvement.

### 🌟 **Ways to Contribute**

- **🐛 Report bugs** - Help us identify and fix issues
- **💡 Suggest features** - Share your ideas for new capabilities  
- **📝 Improve docs** - Help make our documentation even better
- **🔧 Submit code** - Fix bugs, add features, optimize performance
- **🧪 Add examples** - Show how Syda works in your domain
- **⭐ Star the repo** - Help others discover Syda

### 📋 **How to Get Started**

1. **Check our [Contributing Guide](CONTRIBUTING.md)** for detailed instructions
2. **Browse [open issues](https://github.com/syda-ai/syda/issues)** to find something to work on
3. **Join discussions** in our GitHub Issues and Discussions
4. **Fork the repo** and submit your first pull request!

### 🎯 **Good First Issues**

Looking for ways to contribute? Check out issues labeled:
- `good first issue` - Perfect for newcomers
- `help wanted` - We'd especially appreciate help here
- `documentation` - Help improve our docs
- `examples` - Add new use cases and examples

**Every contribution matters - from fixing typos to adding major features!** 🙏

---

**⭐ Star this repo** if Syda helps your workflow • **📖 [Read the docs](https://python.syda.ai)** for detailed guides • **🐛 [Report issues](https://github.com/syda-ai/syda/issues)** to help us improve
