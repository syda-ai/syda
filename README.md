# Synthetic Data Generation Library

A Python-based open-source library for generating synthetic data with AI while preserving referential integrity.

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Core API](#core-api)

  * [Structured Data Generation](#structured-data-generation)
  * [SQLAlchemy Model Integration](#sqlalchemy-model-integration)
  * [Handling Foreign Key Relationships](#handling-foreign-key-relationships)
  * [Automatic Management of Multiple Related Models](#automatic-management-of-multiple-related-models)
  * [Complete CRM Example](#complete-crm-example)
  * [Metadata Enhancement Benefits](#metadata-enhancement-benefits)
  * [Model Selection and Configuration](#model-selection-and-configuration)
  * [Output Options](#output-options)
* [Configuration](#configuration)
* [Contributing](#contributing)
* [License](#license)

## Features

* **Synthetic Data Generation**:

  * Statistical data generation
  * Pattern-based generation
  * Data distribution preservation
  * Synthetic data from various sources
* **Synthetic Data Generation**:

  * Statistical data generation
  * Pattern-based generation
  * Data distribution preservation
  * Synthetic data from various sources
* **Referential Integrity**

  * Automatic foreign key detection and resolution
  * Multi-model dependency analysis
* **Custom Generators**

  * Register column- or type-specific functions for specialized data
* **Open Core**

  * Core functionality under AGPL-3.0
  * Premium UI and SaaS features under commercial license

## Installation

Install the package using pip:

```bash
pip install syda
```

## Quick Start

```python
from syda.structured import SyntheticDataGenerator

generator = SyntheticDataGenerator()
schema = {
    'patient_id': 'number',
    'diagnosis_code': 'icd10_code',
    'email': 'email',
    'visit_date': 'date',
    'notes': 'text'
}
prompt = "Generate realistic synthetic patient records with ICD-10 diagnosis codes, emails, visit dates, and clinical notes."

# Generate and save to CSV
output = generator.generate_data(
    schema=schema,
    prompt=prompt,
    sample_size=15,
    output_path='synthetic_output.csv'
)
print(f"Data saved to {output}")
```

## Core API

### Structured Data Generation

Use simple schema maps or SQLAlchemy models to generate data:

```python
from syda.structured import SyntheticDataGenerator

generator = SyntheticDataGenerator(model='gpt-4-turbo')
# Simple dict schema
records = generator.generate_data(
    schema={'id': 'number', 'name': 'text'},
    prompt='Generate user records',
    sample_size=10
)
```

### SQLAlchemy Model Integration

Pass declarative models directly—docstrings and column metadata inform the prompt:

```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from syda.structured import SyntheticDataGenerator

Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String, comment="Full name of the user")

generator = SyntheticDataGenerator()
generator.generate_data(schema=User, prompt='Generate users', sample_size=5)
```

### Handling Foreign Key Relationships

The library provides robust support for handling foreign key relationships with referential integrity:

1. **Automatic Foreign Key Detection**: Foreign keys are automatically detected from your SQLAlchemy models and assigned the type `'foreign_key'`.
2. **Column-Specific Foreign Key Generators**: Register different generators for each foreign key column when dealing with multiple relationships:

```python
# After generating departments and loading them into departments_df:
def department_id_fk_generator(row, col_name):
    return random.choice(departments_df['id'].tolist())
generator.register_generator('foreign_key', department_id_fk_generator, column_name='department_id')
```

3. **Multi-Step Generation Process**: For related tables, generate parent records first, then use their IDs when generating child records:

```python
# Generate departments first
departments_df = generator.generate_data(schema=Department, prompt='...', sample_size=5)
# Then generate employees with valid department_id references
employees_df = generator.generate_data(schema=Employee, prompt='Generate realistic employee data', sample_size=10)
```

4. **Referential Integrity Preservation**: The foreign key generator samples from actual existing IDs in the parent table, ensuring all references are valid.
5. **Metadata-Enhanced Foreign Keys**: Column comments on foreign key fields are preserved and included in the prompt, helping the LLM understand the relationship context.

### Automatic Management of Multiple Related Models

Simplify multi-table workflows with `generate_related_data`:

```python
results = generator.generate_related_data(
    models=[Customer, Contact, Product, Order, OrderItem],
    prompts={
        "Customer": "Generate diverse customer organizations for a B2B SaaS company.",
        "Product": "Generate cloud software products and services."
    },
    sample_sizes={
        "Customer": 10,
        "Contact": 25,
        "Product": 15,
        "Order": 30,
        "OrderItem": 60
    },
    output_dir="output_data",
    custom_generators={
        "Customer": {"status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"])},
        "Product": {"price": lambda row, col: round(random.uniform(50, 5000), 2)}
    }
)
```

This method:

* Automatically analyzes model dependencies and orders generation.
* Manages foreign keys by sampling valid parent IDs.
* Supports custom generators and preserves referential integrity.

### Complete CRM Example

Here’s a comprehensive example demonstrating `generate_related_data` across five interrelated models, including entity definitions, prompt setup, and data verification:

```python
#!/usr/bin/env python
import random
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Date, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship
from syda.structured import SyntheticDataGenerator

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, comment="Customer organization name")
    industry = Column(String(50), comment="Customer's primary industry")
    website = Column(String(100), comment="Customer's website URL")
    status = Column(String(20), comment="Active, Inactive, Prospect")
    created_at = Column(Date, default=datetime.date.today, comment="Date when added to CRM")
    contacts = relationship("Contact", back_populates="customer")
    orders = relationship("Order", back_populates="customer")

class Contact(Base):
    __tablename__ = 'contacts'
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), comment="Customer this contact belongs to")
    first_name = Column(String(50), comment="Contact's first name")
    last_name = Column(String(50), comment="Contact's last name")
    email = Column(String(100), unique=True, comment="Contact's email address")
    phone = Column(String(20), comment="Contact's phone number")
    position = Column(String(100), comment="Job title or position")
    is_primary = Column(Boolean, default=False, comment="Primary contact flag")
    customer = relationship("Customer", back_populates="contacts")

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, comment="Product name")
    category = Column(String(50), comment="Product category")
    price = Column(Float, comment="Product price in USD")
    description = Column(Text, comment="Product description")
    order_items = relationship("OrderItem", back_populates="product")

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), comment="Customer who placed the order")
    order_date = Column(Date, comment="Date when order was placed")
    status = Column(String(20), comment="Order status: New, Processing, Shipped, Delivered, Cancelled")
    total_amount = Column(Float, comment="Total amount in USD")
    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = 'order_items'
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), comment="Order this item belongs to")
    product_id = Column(Integer, ForeignKey('products.id'), comment="Product in the order")
    quantity = Column(Integer, comment="Quantity ordered")
    unit_price = Column(Float, comment="Unit price at order time")
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


def main():
    generator = SyntheticDataGenerator(model='gpt-4')
    output_dir = 'crm_data'
    prompts = {
        "Customer": "Generate diverse customer organizations for a B2B SaaS company.",
        "Product": "Generate products for a cloud software company.",
        "Order": "Generate realistic orders with appropriate dates and statuses."
    }
    sample_sizes = {"Customer": 10, "Contact": 25, "Product": 15, "Order": 30, "OrderItem": 60}

    results = generator.generate_related_data(
        models=[Customer, Contact, Product, Order, OrderItem],
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=output_dir
    )

    # Referential integrity checks
    print("\n🔍 Verifying referential integrity:")
    if set(results['Contact']['customer_id']).issubset(set(results['Customer']['id'])):
        print("  ✅ All Contact.customer_id values are valid.")
    if set(results['OrderItem']['product_id']).issubset(set(results['Product']['id'])):
        print("  ✅ All OrderItem.product_id values are valid.")
```

## Metadata Enhancement Benefits

* **Richer Context**: Leverages docstrings, comments, and column constraints to enrich prompts.
* **Simpler Prompts**: Less manual specification; model infers details.
* **Constraint Awareness**: Respects `nullable`, `unique`, and length constraints.
* **Custom Generators**: Column-level functions for fine-tuned data.
* **Automatic Docstring Utilization**: Embeds business context from model definitions.

## Model Selection and Configuration

Configure provider, model, temperature, tokens, and proxy:

```python
from syda.schemas import ModelConfig, ProxyConfig
config = ModelConfig(
    provider='openai',
    model_name='gpt-4-turbo',
    temperature=0.9,
    top_p=0.95,
    seed=42,
    max_tokens=1000,
    proxy=ProxyConfig(
        base_url='https://ai-proxy.company.com/v1',
        headers={'X-Company-Auth':'internal-token'},
        params={'team':'data-science'},
        path_format='/proxy/{provider}/completions'
    )
)
generator = SyntheticDataGenerator(model_config=config)
```

## Output Options

* Returns a `pandas.DataFrame` if no `output_path` specified.
* Saves to `.csv` or `.json` when `output_path` ends accordingly.

## Configuration

Set API keys via environment variables or parameters:

```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

Or in code:

```python
generator = SyntheticDataGenerator(
    openai_api_key='...',
    anthropic_api_key='...'
)
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details.
