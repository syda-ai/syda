# Synthetic Data Generation Library

A Python-based open-source library for generating synthetic data with AI while preserving referential integrity. Allowing seamless use of OpenAI, Anthropic (Claude), and other AI models.

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Core API](#core-api)
  * [Structured Data Generation](#structured-data-generation)
  * [SQLAlchemy Model Integration](#sqlalchemy-model-integration)
  * [Handling Foreign Key Relationships](#handling-foreign-key-relationships)
  * [Multiple Schema Definition Formats](#multiple-schema-definition-formats)
    * [SQLAlchemy Models](#1-sqlalchemy-models)
    * [YAML Schema Files](#2-yaml-schema-files)
    * [JSON Schema Files](#3-json-schema-files)
    * [Dictionary-Based Schemas](#4-dictionary-based-schemas)
    * [Foreign Key Definition Methods](#foreign-key-definition-methods)
  * [Automatic Management of Multiple Related Models](#automatic-management-of-multiple-related-models)
    * [Using SQLAlchemy Models](#using-sqlalchemy-models)
    * [Using YAML Schema Files](#using-yaml-schema-files)
    * [Using JSON Schema Files](#using-json-schema-files)
    * [Using Dictionary-Based Schemas](#using-dictionary-based-schemas)
  * [Complete CRM Example](#complete-crm-example)
* [Metadata Enhancement Benefits with SQLAlchemy Models](#metadata-enhancement-benefits-with-sqlalchemy-models)
* [Custom Generators for Domain-Specific Data](#custom-generators-for-domain-specific-data)
* [Unstructured Document Generation](#unstructured-document-generation)
  * [Template-Based Document Generation](#template-based-document-generation)
  * [Template Schema Requirements](#template-schema-requirements)
  * [Supported Template Types](#supported-template-types)
* [Combined Structured and Unstructured Data](#combined-structured-and-unstructured-data)
  * [Connecting Documents to Structured Data](#connecting-documents-to-structured-data)
  * [Schema Dependencies for Documents](#schema-dependencies-for-documents)
  * [Custom Generators for Document Data](#custom-generators-for-document-data)
* [SQLAlchemy Models with Templates](#sqlalchemy-models-with-templates)
* [Model Selection and Configuration](#model-selection-and-configuration)
  * [Basic Configuration](#basic-configuration)
  * [Using Different Model Providers](#using-different-model-providers)
    * [OpenAI Models](#openai-models)
    * [Anthropic Claude Models](#anthropic-claude-models)
    * [Maximum Tokens Parameter](#maximum-tokens-parameter)
    * [Provider-Specific Optimizations](#provider-specific-optimizations)
  * [Advanced: Direct Access to LLM Client](#advanced-direct-access-to-llm-client)
* [Output Options](#output-options)
* [Configuration and Error Handling](#configuration-and-error-handling)
  * [API Keys Management](#api-keys-management)
    * [Environment Variables (Recommended)](#1-environment-variables-recommended)
    * [Direct Initialization](#2-direct-initialization)
  * [Error Handling](#error-handling)
* [Contributing](#contributing)
* [License](#license)

## Features

* **Multi-Provider AI Integration**:

  * Seamless integration with multiple AI providers
  * Support for OpenAI (GPT) and Anthropic (Claude). 
  * Default model is Anthropic Claude model claude-3-5-haiku-20241022
  * Consistent interface across different providers
  * Provider-specific parameter optimization

* **LLM-based Data Generation**:

  * AI-powered schema understanding and data creation
  * Contextually-aware synthetic records
  * Natural language prompt customization
  * Intelligent schema inference

* **SQLAlchemy Integration**:

  * Automatic extraction of model metadata, docstrings and constraints
  * Intelligent column-specific data generation
  * Parameter naming consistency with `sqlalchemy_models`
  
* **Referential Integrity**

  * Automatic foreign key detection and resolution
  * Multi-model dependency analysis through topological sorting
  * Robust handling of related data with referential constraints
  
* **Custom Generators**

  * Register column- or type-specific functions for domain-specific data
  * Contextual generators that adapt to other fields (like ICD-10 codes based on demographics)
  * Weighted distributions for realistic data patterns

* **Open Core**

  * Core functionality under AGPL-3.0

## Installation

Install the package using pip:

```bash
pip install syda
```

## Quick Start

```python
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig

model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192  # Larger value for more complete responses
)

generator = SyntheticDataGenerator(model_config=model_config)

# Define schema for a single table
schemas = {
    'Patient': {
        'patient_id': 'number',
        'diagnosis_code': 'icd10_code',
        'email': 'email',
        'visit_date': 'date',
        'notes': 'text'
    }
}

prompt = "Generate realistic synthetic patient records with ICD-10 diagnosis codes, emails, visit dates, and clinical notes."

# Generate and save to CSV
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts={'Patient': prompt},
    sample_sizes={'Patient': 15},
    output_dir='synthetic_output'
)
print(f"Data saved to synthetic_output/Patient.csv")
```

## Core API

### Structured Data Generation

Use simple schema maps or SQLAlchemy models to generate data:

```python
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig

model_config = ModelConfig(provider='anthropic', model_name='claude-3-5-haiku-20241022')
generator = SyntheticDataGenerator(model_config=model_config)

# Simple dict schema
schemas = {
    'User': {'id': 'number', 'name': 'text'}
}
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts={'User': 'Generate user records'},
    sample_sizes={'User': 10}
)
```

### SQLAlchemy Model Integration

Pass declarative models directly—docstrings and column metadata inform the prompt:

```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig

Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String, comment="Full name of the user")

model_config = ModelConfig(provider='anthropic', model_name='claude-3-5-haiku-20241022')
generator = SyntheticDataGenerator(model_config=model_config)
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[User], 
    prompts={'User': 'Generate users'}, 
    sample_sizes={'User': 5}
)
```

### SQLAlchemy Model Integration

Pass declarative models directly—docstrings and column metadata inform the prompt:

```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig

Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String, comment="Full name of the user")

model_config = ModelConfig(provider='anthropic', model_name='claude-3-5-haiku-20241022')
generator = SyntheticDataGenerator(model_config=model_config)
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[User], 
    prompts={'users': 'Generate users'}, 
    sample_sizes={'users': 5}
)
```

> **Important:** SQLAlchemy models **must** have either `__table__` or `__tablename__` specified. Without one of these attributes, the model cannot be properly processed by the system. The `__tablename__` attribute defines the name of the database table and is used as the schema name when generating data. For example, a model with `__tablename__ = 'users'` will be referenced as 'users' in prompts, sample_sizes, custom generators and the returned results dictionary.


### Handling Foreign Key Relationships

The library provides robust support for handling foreign key relationships with referential integrity:

1. **Automatic Foreign Key Detection**: Foreign keys are automatically detected from your yml, json, dict, SQLAlchemy models and assigned the type `'foreign_key'`.
2. **Manual Column-Specific Foreign Key Generators**: You can also manually define foreign key generators for specific columns as below snippet

```python
# After generating departments and loading them into departments_df:
def department_id_fk_generator(row, col_name):
    return random.choice(departments_df['id'].tolist())
generator.register_generator('foreign_key', department_id_fk_generator, column_name='department_id')
```

3. **Multi-Step Generation Process**: For related tables, generate parent records first, then use their IDs when generating child records:

```python
# Generate departments first, then employees with valid department_id references
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Department, Employee],
    prompts={
        'departments': 'Generate company departments',
        'employees': 'Generate realistic employee data'
    },
    sample_sizes={
        'departments': 5,
        'employees': 10
    }
)

# Access the generated dataframes
departments_df = results['departments']
employees_df = results['employees']
```

4. **Referential Integrity Preservation**: The foreign key generator samples from actual existing IDs in the parent table, ensuring all references are valid.
5. **Metadata-Enhanced Foreign Keys**: Column comments on foreign key fields are preserved and included in the prompt, helping the LLM understand the relationship context.


### Multiple Schema Definition Formats


> **Note:** For detailed information on supported field types and schema format, see the [Schema Reference](schema_reference.md) document.


Syda supports defining your data models in multiple formats, all leading to the same synthetic data generation capabilities. Choose the format that best suits your workflow:

#### 1. SQLAlchemy Models

```python
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    __doc__ = """Customer organization that places orders"""
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, comment="Company name")
    status = Column(String(20), comment="Customer status (Active/Inactive/Prospect)")

class Order(Base):
    __tablename__ = 'orders'
    __doc__ = """Customer order for products or services"""
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    order_date = Column(Date, nullable=False, comment="Date when order was placed")
    total_amount = Column(Float, comment="Total monetary value of the order in USD")

# Generate data from SQLAlchemy models
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Customer, Order],
    prompts={"customers": "Generate tech companies"},
    sample_sizes={"customers": 10, "orders": 30}
)
```

#### 2. YAML Schema Files

```yaml
# customer.yaml
__table_description__: Customer organization that places orders
id:
  type: number
  primary_key: true
name:
  type: text
  max_length: 100
  not_null: true
  description: Company name
status:
  type: text
  max_length: 20
  description: Customer status (Active/Inactive/Prospect)
```

```yaml
# order.yaml
__table_description__: Customer order for products or services
__foreign_keys__:
  customer_id: [Customer, id]
id:
  type: number
  primary_key: true
customer_id:
  type: foreign_key
  not_null: true
  description: Reference to the customer who placed the order
order_date:
  type: date
  not_null: true
  description: Date when order was placed
total_amount:
  type: number
  description: Total monetary value of the order in USD
```

```python
# Generate data from YAML schema files
results = generator.generate_for_schemas(
    schemas={
        'Customer': 'schemas/customer.yaml',
        'Order': 'schemas/order.yaml'
    },
    prompts={'Customer': 'Generate tech companies'},
    sample_sizes={'Customer': 10, 'Order': 30}
)
```

#### 3. JSON Schema Files

```json
// customer.json
{
  "__table_description__": "Customer organization that places orders",
  "id": {
    "type": "number",
    "primary_key": true
  },
  "name": {
    "type": "text",
    "max_length": 100,
    "not_null": true,
    "description": "Company name"
  },
  "status": {
    "type": "text",
    "max_length": 20,
    "description": "Customer status (Active/Inactive/Prospect)"
  }
}
```

```json
// order.json
{
  "__table_description__": "Customer order for products or services",
  "__foreign_keys__": {
    "customer_id": ["Customer", "id"]
  },
  "id": {
    "type": "number",
    "primary_key": true
  },
  "customer_id": {
    "type": "foreign_key",
    "not_null": true,
    "description": "Reference to the customer who placed the order"
  },
  "order_date": {
    "type": "date",
    "not_null": true,
    "description": "Date when order was placed"
  },
  "total_amount": {
    "type": "number",
    "description": "Total monetary value of the order in USD"
  }
}
```

```python
# Generate data from JSON schema files
results = generator.generate_for_schemas(
    schemas={
        'Customer': 'schemas/customer.json',
        'Order': 'schemas/order.json'
    },
    prompts={'Customer': 'Generate tech companies'},
    sample_sizes={'Customer': 10, 'Order': 30}
)
```

#### 4. Dictionary-Based Schemas

```python
# Define schemas directly as dictionaries
schemas = {
    'Customer': {
        '__table_description__': 'Customer organization that places orders',
        'id': {'type': 'number', 'primary_key': True},
        'name': {
            'type': 'text',
            'max_length': 100,
            'not_null': True,
            'description': 'Company name'
        },
        'status': {
            'type': 'text',
            'max_length': 20,
            'description': 'Customer status (Active/Inactive/Prospect)'
        }
    },
    'Order': {
        '__table_description__': 'Customer order for products or services',
        '__foreign_keys__': {
            'customer_id': ['Customer', 'id']
        },
        'id': {'type': 'number', 'primary_key': True},
        'customer_id': {
            'type': 'foreign_key',
            'not_null': True,
            'description': 'Reference to the customer who placed the order'
        },
        'order_date': {
            'type': 'date',
            'not_null': True,
            'description': 'Date when order was placed'
        },
        'total_amount': {
            'type': 'number',
            'description': 'Total monetary value of the order in USD'
        }
    }
}

# Generate data from dictionary schemas
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts={'Customer': 'Generate tech companies'},
    sample_sizes={'Customer': 10, 'Order': 30}
)
```

#### Foreign Key Definition Methods

There are three ways to define foreign key relationships:

1. Using the `__foreign_keys__` special section in a schema:
   ```python
   "__foreign_keys__": {
       "customer_id": ["Customer", "id"]
   }
   ```

2. Using field-level references with type and references properties:
   ```python
   "order_id": {
       "type": "foreign_key",
       "references": {
           "schema": "Order",
           "field": "id"
       }
   }
   ```

3. Using type-based detection with naming conventions:
   ```python
   "customer_id": "foreign_key"
   ```
   (The system will attempt to infer the relationship based on naming conventions)

### Automatic Management of Multiple Related Models

#### Using SQLAlchemy Models

Simplify multi-table workflows with `generate_for_sqlalchemy_models`:

```python
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import random
from syda.generate import SyntheticDataGenerator

Base = declarative_base()

# Customer model
class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    industry = Column(String(50))
    status = Column(String(20))
    contacts = relationship("Contact", back_populates="customer")
    orders = relationship("Order", back_populates="customer")

# Contact model with foreign key to Customer
class Contact(Base):
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(120), nullable=False)
    phone = Column(String(20))
    customer = relationship("Customer", back_populates="contacts")

# Product model
class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    price = Column(Float, nullable=False)
    order_items = relationship("OrderItem", back_populates="product")

# Order model with foreign key to Customer
class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    order_date = Column(Date, nullable=False)
    total_amount = Column(Float)
    customer = relationship("Customer", back_populates="orders")
    order_items = relationship("OrderItem", back_populates="order")

# OrderItem model with foreign keys to Order and Product
class OrderItem(Base):
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    order = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")

# Initialize generator
generator = SyntheticDataGenerator()

# Generate data for all models in one call
results = generator.generate_for_sqlalchemy_models(
    models=[Customer, Contact, Product, Order, OrderItem],
    prompts={
        "customers": "Generate diverse customer organizations for a B2B SaaS company.",
        "contacts": "Generate cloud software products and services."
    },
    sample_sizes={
        "customers": 10,
        "contacts": 25,
        "products": 15,
        "orders": 30,
        "order_items": 60
    },
    custom_generators={
        "customers": {
            # Ensure a specific distribution of customer statuses for business reporting
            "status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"]),
        },
        "products": {
            # Ensure product categories match your specific business domains
            "category": lambda row, col: random.choice([
                "Cloud Infrastructure", "Business Intelligence", "Security Services",
                "Data Analytics", "Custom Development", "Support Package", "API Services"
            ])
        },
    }
)
```

#### Using YAML Schema Files

The same relationship management is available with YAML schemas:

```yaml
# customer.yaml
__table_name__: customers
__description__: Customer organizations

id:
  type: integer
  constraints:
    primary_key: true
    not_null: true

name:
  type: string
  constraints:
    not_null: true
    max_length: 100

industry:
  type: string
  constraints:
    max_length: 50

status:
  type: string
  constraints:
    max_length: 20
```

```yaml
# contact.yaml
__table_name__: contacts
__description__: Customer contacts
__foreign_keys__:
  customer_id: [customers, id]

id:
  type: integer
  constraints:
    primary_key: true
    not_null: true

customer_id:
  type: integer
  constraints:
    not_null: true

name:
  type: string
  constraints:
    not_null: true
    max_length: 100

email:
  type: string
  constraints:
    not_null: true
    max_length: 120

phone:
  type: string
  constraints:
    max_length: 20
```

```yaml
# order.yaml
__table_name__: orders
__description__: Customer orders
__foreign_keys__:
  customer_id: [customers, id]

id:
  type: integer
  constraints:
    primary_key: true
    not_null: true

customer_id:
  type: integer
  constraints:
    not_null: true

order_date:
  type: string
  format: date
  constraints:
    not_null: true

total_amount:
  type: number
  format: float
```

```python
# Generate data for multiple related tables with YAML schemas
results = generator.generate_for_schemas(
    schemas={
        'Customer': 'schemas/customer.yaml',
        'Contact': 'schemas/contact.yaml',
        'Product': 'schemas/product.yaml',
        'Order': 'schemas/order.yaml',
        'OrderItem': 'schemas/order_item.yaml'
    },
    prompts={
        "Customer": "Generate diverse customer organizations for a B2B SaaS company.",
        "Product": "Generate cloud software products and services."
    },
    sample_sizes={
        "Customer": 10,
        "Contact": 20,
        "Product": 15,
        "Order": 30,
        "OrderItem": 60
    }
)
```

#### Using JSON Schema Files

JSON schema files offer the same capabilities:

```json
// customer.json
{
  "__table_name__": "customers",
  "__description__": "Customer organizations",
  "id": {
    "type": "integer",
    "constraints": {
      "primary_key": true,
      "not_null": true
    }
  },
  "name": {
    "type": "string",
    "constraints": {
      "not_null": true,
      "max_length": 100
    }
  },
  "industry": {
    "type": "string",
    "constraints": {
      "max_length": 50
    }
  },
  "status": {
    "type": "string",
    "constraints": {
      "max_length": 20
    }
  }
}
```

```json
// contact.json
{
  "__table_name__": "contacts",
  "__description__": "Customer contacts",
  "__foreign_keys__": {
    "customer_id": ["customers", "id"]
  },
  "id": {
    "type": "integer",
    "constraints": {
      "primary_key": true,
      "not_null": true
    }
  },
  "customer_id": {
    "type": "integer",
    "constraints": {
      "not_null": true
    }
  },
  "name": {
    "type": "string",
    "constraints": {
      "not_null": true,
      "max_length": 100
    }
  },
  "email": {
    "type": "string",
    "constraints": {
      "not_null": true,
      "max_length": 120
    }
  },
  "phone": {
    "type": "string",
    "constraints": {
      "max_length": 20
    }
  }
}
```

```json
// order.json
{
  "__table_name__": "orders",
  "__description__": "Customer orders",
  "__foreign_keys__": {
    "customer_id": ["customers", "id"]
  },
  "id": {
    "type": "integer",
    "constraints": {
      "primary_key": true,
      "not_null": true
    }
  },
  "customer_id": {
    "type": "integer",
    "constraints": {
      "not_null": true
    }
  },
  "order_date": {
    "type": "string",
    "format": "date",
    "constraints": {
      "not_null": true
    }
  },
  "total_amount": {
    "type": "number",
    "format": "float"
  }
}
```

```python
# Generate data for multiple related tables with JSON schemas
results = generator.generate_for_schemas(
    schemas={
        'Customer': 'schemas/customer.json',
        'Contact': 'schemas/contact.json',
        'Product': 'schemas/product.json',
        'Order': 'schemas/order.json',
        'OrderItem': 'schemas/order_item.json'
    },
    prompts={
        "Customer": "Generate diverse customer organizations for a B2B SaaS company.",
        "Product": "Generate cloud software products and services."
    },
    sample_sizes={
        "Customer": 10,
        "Contact": 20,
        "Product": 15,
        "Order": 30,
        "OrderItem": 60
    }
)
```

#### Using Dictionary-Based Schemas

Similar relationship management works with dictionary schemas:

```python
# Define schemas as Python dictionaries
schemas = {
    'Customer': {
        '__table_name__': 'customers',
        '__description__': 'Customer organizations',
        'id': {
            'type': 'integer',
            'constraints': {'primary_key': True, 'not_null': True}
        },
        'name': {
            'type': 'string',
            'constraints': {'not_null': True, 'max_length': 100}
        },
        'industry': {
            'type': 'string',
            'constraints': {'max_length': 50}
        },
        'status': {
            'type': 'string',
            'constraints': {'max_length': 20}
        }
    },
    'Contact': {
        '__table_name__': 'contacts',
        '__description__': 'Customer contacts',
        '__foreign_keys__': {
            'customer_id': ['customers', 'id']
        },
        'id': {
            'type': 'integer',
            'constraints': {'primary_key': True, 'not_null': True}
        },
        'customer_id': {
            'type': 'integer',
            'constraints': {'not_null': True}
        },
        'name': {
            'type': 'string',
            'constraints': {'not_null': True, 'max_length': 100}
        },
        'email': {
            'type': 'string',
            'constraints': {'not_null': True, 'max_length': 120}
        },
        'phone': {
            'type': 'string',
            'constraints': {'max_length': 20}
        }
    },
    'Order': {
        '__table_name__': 'orders',
        '__description__': 'Customer orders',
        '__foreign_keys__': {
            'customer_id': ['customers', 'id']
        },
        'id': {
            'type': 'integer',
            'constraints': {'primary_key': True, 'not_null': True}
        },
        'customer_id': {
            'type': 'integer',
            'constraints': {'not_null': True}
        },
        'order_date': {
            'type': 'string',
            'format': 'date',
            'constraints': {'not_null': True}
        },
        'total_amount': {
            'type': 'number',
            'format': 'float'
        }
    }
}

# Generate data for dictionary schemas
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts={
        'Customer': 'Generate diverse customer organizations for a B2B SaaS company.'
    },
    sample_sizes={
        'Customer': 10,
        'Contact': 20,
        'Order': 30
    }
)
```

In all cases, the generator will:
1. Analyze relationships between models/schemas
2. Determine the correct generation order using topological sorting
3. Generate parent tables first
4. Use existing primary keys when populating foreign keys in child tables
5. Maintain referential integrity across the entire dataset


### Complete CRM Example

Here’s a comprehensive example demonstrating `generate_for_sqlalchemy_models` across five interrelated models, including entity definitions, prompt setup, and data verification:

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
        "customers": "Generate diverse customer organizations for a B2B SaaS company.",
        "products": "Generate products for a cloud software company.",
        "orders": "Generate realistic orders with appropriate dates and statuses."
    }
    sample_sizes = {"customers": 10, "contacts": 25, "products": 15, "orders": 30, "order_items": 60}

    results = generator.generate_for_sqlalchemy_models(
        sqlalchemy_models=[Customer, Contact, Product, Order, OrderItem],
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

## Metadata Enhancement Benefits with SQLAlchemy Models

* **Richer Context**: Leverages docstrings, comments, and column constraints to enrich prompts.
* **Simpler Prompts**: Less manual specification; model infers details.
* **Constraint Awareness**: Respects `nullable`, `unique`, and length constraints.
* **Custom Generators**: Column-level functions for fine-tuned data.
* **Automatic Docstring Utilization**: Embeds business context from model definitions.


## Unstructured Document Generation

SYDA can generate realistic unstructured documents such as PDF reports, letters, and forms based on templates. This is useful for applications that require document generation with synthetic data.

For complete examples, see the [examples/unstructured_only](examples/unstructured_only) directory, which includes healthcare document generation samples.

### Template-Based Document Generation

Create template-based document schemas by specifying template fields in your schema:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Initialize generator 
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)

# Define template-based schemas
schemas = {
    'MedicalReport': 'schemas/medical_report.yml',
    'LabResult': 'schemas/lab_result.yml'
}
```

Here's an example of a medical report template schema:

```yaml
# Medical report template schema (medical_report.yml)
__template__: true
__description__: Medical report template for patient visits
__name__: MedicalReport
__foreign_keys__: {}
__template_source__: templates/medical_report_template.html
__input_file_type__: html
__output_file_type__: pdf

# Patient information
patient_id:
  type: string
  format: uuid

patient_name:
  type: string

date_of_birth:
  type: string
  format: date

visit_date:
  type: string
  format: date-time

chief_complaint:
  type: string

medical_history:
  type: string

# Vital signs
blood_pressure:
  type: string

heart_rate:
  type: integer

respiratory_rate:
  type: integer

temperature:
  type: number

oxygen_saturation:
  type: integer

# Clinical information
assessment:
  type: string

# Generate data and PDF documents
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={
        'MedicalReport': 5,
        'LabResult': 5
    },
    prompts={
        'MedicalReport': 'Generate synthetic medical reports for patients',
        'LabResult': 'Generate synthetic laboratory test results for patients'
    },
    output_dir="output"
)
```

### Template Schema Requirements

Template-based schemas must include these special fields:

```yaml
__template__: true
__template_source__: /path/to/template.html
__input_file_type__: html
__output_file_type__: pdf
```

The template file (like HTML) includes variable placeholders that get replaced with generated data. Here's an example of a Jinja2 HTML template for medical reports corresponding to the schema above:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Medical Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 20px;
        }
        .section-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>MEDICAL REPORT</h1>
    </div>
    
    <div class="section">
        <div class="section-title">PATIENT INFORMATION</div>
        <p>
            <strong>Patient ID:</strong> {{ patient_id }}<br>
            <strong>Name:</strong> {{ patient_name }}<br>
            <strong>Date of Birth:</strong> {{ date_of_birth }}
        </p>
    </div>
    
    <div class="section">
        <div class="section-title">VISIT INFORMATION</div>
        <p>
            <strong>Visit Date:</strong> {{ visit_date }}<br>
            <strong>Chief Complaint:</strong> {{ chief_complaint }}
        </p>
    </div>
    
    <div class="section">
        <div class="section-title">MEDICAL HISTORY</div>
        <p>{{ medical_history }}</p>
    </div>
    
    <div class="section">
        <div class="section-title">VITAL SIGNS</div>
        <p>
            <strong>Blood Pressure:</strong> {{ blood_pressure }}<br>
            <strong>Heart Rate:</strong> {{ heart_rate }} bpm<br>
            <strong>Respiratory Rate:</strong> {{ respiratory_rate }} breaths/min<br>
            <strong>Temperature:</strong> {{ temperature }}°F<br>
            <strong>Oxygen Saturation:</strong> {{ oxygen_saturation }}%
        </p>
    </div>
    
    <div class="section">
        <div class="section-title">ASSESSMENT</div>
        <p>{{ assessment }}</p>
    </div>
</body>
</html>
```

As you can see, the template uses Jinja2's `{{ variable_name }}` syntax to insert the data from the generated schema fields into the HTML document.

### Supported Template Types

- HTML → PDF: Best supported with complete styling control
- HTML → HTML: Simple text formatting

More template formats will be supported in next versions

## Combined Structured and Unstructured Data

SYDA excels at generating both structured data (tables/databases) and unstructured content (documents) in a coordinated way.

For working examples, see the [examples/structured_and_unstructured](examples/structured_and_unstructured) directory, which contains retail receipt generation and CRM document examples.


### Connecting Documents to Structured Data

You can create relationships between document schemas and structured data schemas:

```python
from syda.generate import SyntheticDataGenerator

generator = SyntheticDataGenerator()

# Define both structured and template-based schemas
schemas = {
    'Customer': 'schemas/customer.yml',            # Structured data
    'Product': 'schemas/product.yml',              # Structured data
    'Transaction': 'schemas/transaction.yml',      # Structured data
    'Receipt': 'schemas/receipt.yml'               # Template-based document
}
```

Here's what a structured data schema for a `Customer` might look like:

```yaml
# Customer schema (customer.yml)
__table_name__: Customer
__description__: Retail customers

id:
  type: integer
  description: Unique customer ID
  constraints:
    primary_key: true
    not_null: true
    min: 1

first_name:
  type: string
  description: Customer's first name
  constraints:
    not_null: true
    length: 50

last_name:
  type: string
  description: Customer's last name
  constraints:
    not_null: true
    length: 50
    
email:
  type: email
  description: Customer's email address
  constraints:
    not_null: true
    unique: true
    length: 100
```

And here's a template-based document schema for a `Receipt` that references the structured data:

```yaml
# Receipt template schema (receipt.yml)
__template__: true
__description__: Retail receipt template
__name__: Receipt
__depends_on__: [Product, Transaction, Customer]
__foreign_keys__:
  customer_name: [Customer, first_name]
  
__template_source__: templates/receipt.html
__input_file_type__: html
__output_file_type__: pdf

# Receipt header
store_name:
  type: string
  length: 50
  description: Name of the retail store

store_address:
  type: address
  length: 150
  description: Full address of the store

# Receipt details
receipt_number:
  type: string
  pattern: '^RCP-\d{8}$'
  length: 12
  description: Unique receipt identifier

# Product purchase details
items:
  type: array
  description: "List of purchased items with product details"


# Generate everything - maintains relationships between structured and document data
results = generator.generate_for_schemas(
    schemas=schemas,
    output_dir="output"
)

# Results include both DataFrames and generated documents
customers_df = results['Customer']
receipts_df = results['Receipt']     # Contains metadata about generated documents
```

### Schema Dependencies for Documents

Template schemas can specify dependencies on structured schemas:

```yaml
# Receipt template schema (receipt.yml)
__template__: true
__name__: Receipt
__depends_on__: [Product, Transaction, Customer]
__foreign_keys__:
  customer_id: [Customer, id]
__template_source__: templates/receipt.html
__input_file_type__: html
__output_file_type__: pdf
```

This ensures that dependent structured data is generated first, and related documents can reference that data.

Here's an example of a receipt HTML template that uses data from both the receipt schema and the related structured data:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Receipt</title>
    <style>
        body {
            font-family: 'Courier New', Courier, monospace;
            font-size: 12px;
            line-height: 1.3;
            max-width: 380px;
            margin: 0 auto;
            padding: 10px;
        }
        .header, .footer {
            text-align: center;
            margin-bottom: 10px;
        }
        .items-table {
            width: 100%;
            margin-bottom: 10px;
        }
        .totals {
            width: 100%;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="store-name">{{ store_name }}</div>
        <div>{{ store_address }}</div>
        <div>Tel: {{ store_phone }}</div>
    </div>

    <div class="receipt-details">
        <div>
            <div>Receipt #: {{ receipt_number }}</div>
            <div>Date: {{ transaction_date }}</div>
            <div>Time: {{ transaction_time }}</div>
        </div>
    </div>

    <div class="customer-info">
        <div>Customer: {{ customer_name }}</div>
        <div>Cust ID: {{ customer_id }}</div>
    </div>

    <!-- This iterates through items array generated by the custom generator -->
    <table class="items-table">
        <thead>
            <tr>
                <th>Item</th>
                <th>Qty</th>
                <th>Price</th>
                <th>Total</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td>{{ item.product_name }}<br><small>SKU: {{ item.sku }}</small></td>
                <td>{{ item.quantity }}</td>
                <td>${{ "%.2f"|format(item.unit_price) }}</td>
                <td>${{ "%.2f"|format(item.item_total) }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <table class="totals">
        <tr>
            <td>Subtotal:</td>
            <td>${{ "%.2f"|format(subtotal) }}</td>
        </tr>
        <tr>
            <td>Tax ({{ "%.2f"|format(tax_rate) }}%):</td>
            <td>${{ "%.2f"|format(tax_amount) }}</td>
        </tr>
        <tr>
            <td>TOTAL:</td>
            <td>${{ "%.2f"|format(total) }}</td>
        </tr>
    </table>

    <div class="payment-info">
        <div>Payment Method: {{ payment_method }}</div>
    </div>

    <div class="thank-you">
        Thank you for shopping with us!
    </div>
</body>
</html>
```

Note the use of Jinja2's `{% for item in items %}...{% endfor %}` loop to iterate through the array of items that was generated with our custom generator.

### Custom Generators for Document Data

For advanced use cases, you can define custom generators to map structured data into document fields:

```python
def generate_receipt_items(row, col_name=None, parent_dfs=None):
    """Generate receipt line items based on transaction and product data."""
    items = []
    if parent_dfs and 'Product' in parent_dfs and 'Transaction' in parent_dfs:
        products_df = parent_dfs['Product']
        transactions_df = parent_dfs['Transaction']
        
        # Find transactions for this customer
        customer_transactions = transactions_df[transactions_df['customer_id'] == row['customer_id']]
        
        # Add products from transactions to receipt
        for _, tx in customer_transactions.iterrows():
            product = products_df[products_df['id'] == tx['product_id']].iloc[0]
            items.append({
                "product_name": product['name'],
                "quantity": tx['quantity'],
                "unit_price": product['price'],
                "item_total": tx['quantity'] * product['price']
            })
    return items

# Register the custom generator
generator.register_generator('array', generate_receipt_items, column_name='items')
```

The `parent_dfs` parameter gives access to all previously generated structured data, allowing you to create rich, interconnected documents.


## SQLAlchemy Models with Templates

You can also use SQLAlchemy models to define both your structured data schema and template-based documents. This approach is great for applications that already use SQLAlchemy ORM:

```python
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from syda.templates import SydaTemplate

Base = declarative_base()

# Regular structured SQLAlchemy model
class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    industry = Column(String(50))
    annual_revenue = Column(Float)
    website = Column(String(100))
    
    # Relationships
    opportunities = relationship("Opportunity", back_populates="customer")

# Another structured model
class Opportunity(Base):
    __tablename__ = 'opportunities'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    description = Column(Text)
    
    # Relationships
    customer = relationship("Customer", back_populates="opportunities")

# Template model
class ProposalDocument(Base):
    __tablename__ = 'proposal_documents'
    
    # Special template attributes
    __template__ = True
    __depends_on__ = ['Opportunity']  # This template depends on the Opportunity model
    
    # Template source configuration
    __template_source__ = 'templates/proposal.html'
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    # Fields needed for the template (these become columns in the generated data)
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False)
    title = Column(String(200))
    customer_name = Column(String(100), ForeignKey('customers.name'))
    opportunity_value = Column(Float, ForeignKey('opportunities.value'))
    proposed_solutions = Column(Text)
```

Then generate all data in one call:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Initialize generator
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)

# Generate all data at once
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Customer, Opportunity, ProposalDocument],
    sample_sizes={'customers': 5, 'opportunities': 8, 'proposal_documents': 3},
    output_dir="output"
)
```

The example above demonstrates:
1. Regular SQLAlchemy models for structured data (Customer, Opportunity)
2. A template model (ProposalDocument)
3. Foreign key relationships between the template and structured models
4. Generating everything together with `generate_for_sqlalchemy_models`


## Model Selection and Configuration

Syda currently supports two AI providers: OpenAI and Anthropic (Claude).



### Basic Configuration

Configure provider, model, temperature, tokens, and proxy settings using the `ModelConfig` class:

```python
from syda.schemas import ModelConfig, ProxyConfig

# Create a model configuration
config = ModelConfig(
    provider='openai',  # Choose from: 'openai', 'anthropic', etc.
    model_name='gpt-4-turbo',  # Model name for the selected provider
    temperature=0.7,    # Controls randomness (0.0-1.0)
    top_p=0.95,         # Nucleus sampling parameter
    seed=42,            # For reproducible outputs (provider-specific)
    max_tokens=4000,    # Maximum response length (default: 4000)
    proxy=ProxyConfig(  # Optional proxy configuration
        base_url='https://ai-proxy.company.com/v1',
        headers={'X-Company-Auth':'internal-token'},
        params={'team':'data-science'}
    )
)

# Initialize generator with the configuration
generator = SyntheticDataGenerator(model_config=config)
```

### Using Different Model Providers

The library currently supports OpenAI and Anthropic (Claude) models and allows you to easily switch between these providers while maintaining a consistent interface.

#### OpenAI Models

```python
# Default configuration - uses OpenAI's GPT-4 if no model_config provided
default_generator = SyntheticDataGenerator()

# Explicitly configure for GPT-3.5 Turbo (faster and more cost-effective)
openai_config = ModelConfig(
    provider='openai',
    model_name='gpt-3.5-turbo',  # You can also use 'gpt-3.5-turbo-1106' for better JSON handling
    temperature=0.7,
    response_format={"type": "json_object"}  # Forces JSON response format (GPT models)
)
gpt35_generator = SyntheticDataGenerator(model_config=openai_config)

# Generate data with specific model configuration
data = gpt35_generator.generate_data(
    schema={'product_id': 'number', 'product_name': 'text', 'price': 'number'},
    prompt="Generate electronic product data with prices between $500-$2000",
    sample_size=10
)
```

#### Anthropic Claude Models

```python
# Configure for Claude (requires ANTHROPIC_API_KEY environment variable)
claude_config = ModelConfig(
    provider='anthropic',
    model_name='claude-3-sonnet-20240229',  # Available models: claude-3-opus, claude-3-sonnet, claude-3-haiku
    temperature=0.7,
    max_tokens=2000  # Claude can sometimes need more tokens for structured output
)
claude_generator = SyntheticDataGenerator(model_config=claude_config)

# Generate data with Claude
data = claude_generator.generate_data(
    schema={'product_id': 'number', 'product_name': 'text', 'price': 'number', 'description': 'text'},
    prompt="Generate luxury product data with realistic prices over $1000",
    sample_size=5
)
```

#### Maximum Tokens Parameter

The library now uses a default of 4000 tokens for `max_tokens` to ensure complete responses with all expected columns. This helps prevent incomplete data generation issues.

```python
# Override the default max_tokens setting
config = ModelConfig(
    provider="openai",
    model_name="gpt-4",
    max_tokens=8000,  # Increase for very complex schemas or large sample sizes
    temperature=0.7
)
```

When generating complex data or data with many columns, consider increasing this value if you notice missing columns in your generated data.

#### Provider-Specific Optimizations

Each AI provider has different strengths and parameter requirements. The library automatically handles most of the differences, but you can optimize for specific providers:

```python
# OpenAI-specific optimization
openai_optimized = ModelConfig(
    provider='openai',
    model_name='gpt-4-turbo',
    temperature=0.7,
    response_format={"type": "json_object"},  # Only works with OpenAI
    seed=42  # For reproducible outputs
)

# Anthropic-specific optimization
anthropic_optimized = ModelConfig(
    provider='anthropic',
    model_name='claude-3-opus-20240229',
    temperature=0.7,
    system="You are a synthetic data generator that creates realistic, high-quality datasets based on the provided schema."  # System prompt works best with Anthropic
)
```

### Advanced: Direct Access to LLM Client

For advanced use cases, you can access the underlying LLM client directly for additional control:

```python
from syda.llm import create_llm_client

# Create a standalone LLM client
llm_client = create_llm_client(
    model_config=ModelConfig(
        provider='anthropic', 
        model_name='claude-3-opus-20240229'
    ),
    # API key is optional if set in environment variables
    anthropic_api_key="your_api_key"  
)

# Define a Pydantic model for structured output
from pydantic import BaseModel
from typing import List

class Book(BaseModel):
    title: str
    author: str
    year: int
    genre: str
    pages: int

class BookCollection(BaseModel):
    books: List[Book]

# Use the client for structured responses
books = llm_client.client.chat.completions.create(
    model="claude-3-opus-20240229",
    response_model=BookCollection,  # Automatically parses the response to this model
    messages=[{"role": "user", "content": "Generate 5 fictional sci-fi books."}]
)

# Access the structured data directly
for book in books.books:
    print(f"{book.title} by {book.author} ({book.year}) - {book.pages} pages")
```

This approach gives you direct control over the client while still providing structured data extraction capabilities.

## Output Options

Syda offers flexible output options to suit different use cases:

### Multiple Schema Generation

When generating data for multiple schemas using `generate_for_schemas` or `generate_for_sqlalchemy_models`, you can specify an output directory and format:

```python
# Generate and save data to CSV files (default)
results = generator.generate_for_schemas(
    schemas=schemas,
    output_dir="output_directory",
    output_format="csv"  # Default format
)

# Generate and save data to JSON files
results = generator.generate_for_schemas(
    schemas=schemas,
    output_dir="output_directory",
    output_format="json"
)
```

Each schema will be saved to a separate file with the schema name as the filename. For example:

* CSV format: `output_directory/customer.csv`, `output_directory/order.csv`, etc.
* JSON format: `output_directory/customer.json`, `output_directory/order.json`, etc.

The `results` dictionary will still contain all generated DataFrames, so you can both save to files and work with the data directly in your code.


## Configuration and Error Handling

### API Keys Management

You can provide appropriate API keys based on the provider you're using. There are two recommended ways to manage API keys:

#### 1. Environment Variables (Recommended)

Set API keys via environment variables:

```bash
# For OpenAI models
export OPENAI_API_KEY=your_openai_key

# For Anthropic models
export ANTHROPIC_API_KEY=your_anthropic_key

# For other providers, set the appropriate environment variables
```

You can also use a `.env` file in your project root and load it with:

```python
from dotenv import load_dotenv
load_dotenv()  # This loads API keys from .env file
```

#### 2. Direct Initialization

Provide API keys when initializing the generator:

```python
# With explicit model configuration
generator = SyntheticDataGenerator(
    model_config=ModelConfig(provider='openai', model_name='gpt-4'),
    openai_api_key="your_openai_key",      # Only needed for OpenAI models
    anthropic_api_key="your_anthropic_key"  # Only needed for Anthropic models
)
```


### Error Handling

Syda's error handling has been improved to provide more useful feedback when data generation fails. The library now:

1. **Raises Explicit Exceptions**: When data generation fails rather than returning random data
2. **Provides Detailed Error Messages**: Explaining what went wrong and potential fixes
3. **Validates Output Structure**: Ensures generated data matches the expected schema

Example error handling:

```python
try:
    data = generator.generate_data(
        schema=YourModel,
        prompt="Generate synthetic data...",
        sample_size=10
    )
    # Process the data...
except ValueError as e:
    print(f"Data generation failed: {str(e)}")
    # Implement fallback strategy or retry with different parameters
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

## License

See [LICENSE](LICENSE) for details.
