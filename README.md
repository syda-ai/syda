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
  * [Complete CRM Example](#complete-crm-example)
* [Metadata Enhancement Benefits with SQLAlchemy Models](#metadata-enhancement-benefits-with-sqlalchemy-models)
* [Custom Generators for Domain-Specific Data](#custom-generators-for-domain-specific-data)
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
# Generate departments first, then employees with valid department_id references
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Department, Employee],
    prompts={
        'Department': 'Generate company departments',
        'Employee': 'Generate realistic employee data'
    },
    sample_sizes={
        'Department': 5,
        'Employee': 10
    }
)

# Access the generated dataframes
departments_df = results['Department']
employees_df = results['Employee']
```

4. **Referential Integrity Preservation**: The foreign key generator samples from actual existing IDs in the parent table, ensuring all references are valid.
5. **Metadata-Enhanced Foreign Keys**: Column comments on foreign key fields are preserved and included in the prompt, helping the LLM understand the relationship context.

### Multiple Schema Definition Formats

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
    prompts={"Customer": "Generate tech companies"},
    sample_sizes={"Customer": 10, "Order": 30}
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

Simplify multi-table workflows with `generate_for_sqlalchemy_models`:

```python
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Customer, Contact, Product, Order, OrderItem],
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
        "Customer": "Generate diverse customer organizations for a B2B SaaS company.",
        "Product": "Generate products for a cloud software company.",
        "Order": "Generate realistic orders with appropriate dates and statuses."
    }
    sample_sizes = {"Customer": 10, "Contact": 25, "Product": 15, "Order": 30, "OrderItem": 60}

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

## Custom Generators for Domain-Specific Data

For specialized domains like healthcare, finance, or logistics, you can register custom generators to produce highly accurate and domain-compliant synthetic data. Here's an example with healthcare insurance data:

```python
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
import datetime
import random
from syda.structured import SyntheticDataGenerator

Base = declarative_base()

class Patient(Base):
    """Healthcare patient record with demographic and insurance information."""
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    member_id = Column(String(20), unique=True, comment="Insurance member identifier")
    first_name = Column(String(50), comment="Patient's first name")
    last_name = Column(String(50), comment="Patient's last name")
    birth_date = Column(Date, comment="Patient's date of birth")
    gender = Column(String(10), comment="Patient's gender identity")
    plan_type = Column(String(20), comment="Insurance plan type (HMO, PPO, etc.)")

class Claim(Base):
    """Healthcare insurance claim record for patient services."""
    __tablename__ = 'claims'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), comment="Reference to patient")
    claim_number = Column(String(20), unique=True, comment="Unique claim identifier")
    date_of_service = Column(Date, comment="Date medical service was provided")
    primary_diagnosis = Column(String(10), comment="Primary ICD-10 diagnosis code")
    procedure_code = Column(String(10), comment="CPT procedure code")
    billed_amount = Column(Float, comment="Amount billed for service in USD")
    allowed_amount = Column(Float, comment="Amount allowed by insurance in USD")
    status = Column(String(20), comment="Claim status (Pending, Approved, Denied, etc.)")

# Create the generator instance
generator = SyntheticDataGenerator()

# Generate patients first
patients_df = generator.generate_data(
    schema=Patient,
    prompt="Generate realistic patient demographic data for a health insurance company.",
    sample_size=50,
    output_path="patients.csv"
)

# Register custom generators for domain-specific fields

# 1. Custom generator for contextually relevant ICD-10 diagnosis codes
def icd10_generator(row, col_name):
    # Get procedure code and patient age (calculated from birth_date)
    procedure = row.get("procedure_code", "")
    try:
        birth_date = datetime.datetime.strptime(row.get("birth_date", ""), "%Y-%m-%d").date()
        age = (datetime.date.today() - birth_date).days // 365
    except (ValueError, TypeError):
        # Default to middle-aged adult if birth_date is missing or invalid
        age = 45
    
    gender = row.get("gender", "").lower()
    
    # Map of diagnosis codes by category
    diagnosis_codes = {
        # Cardiovascular conditions (more common in older patients)
        "cardiovascular": [
            ("I10", "Essential hypertension"),
            ("I25.10", "Atherosclerotic heart disease"),
            ("I48.91", "Atrial fibrillation"),
            ("I50.9", "Heart failure, unspecified")
        ],
        # Respiratory conditions
        "respiratory": [
            ("J45.909", "Unspecified asthma"),
            ("J02.9", "Acute pharyngitis"),
            ("J40", "Bronchitis, not specified as acute or chronic"),
            ("J06.9", "Acute upper respiratory infection")
        ],
        # Musculoskeletal conditions 
        "musculoskeletal": [
            ("M54.5", "Low back pain"),
            ("M25.511", "Pain in right shoulder"),
            ("M17.9", "Osteoarthritis of knee"),
            ("M79.1", "Myalgia (muscle pain)")
        ],
        # Endocrine conditions
        "endocrine": [
            ("E11.9", "Type 2 diabetes without complications"),
            ("E78.5", "Hyperlipidemia"),
            ("E03.9", "Hypothyroidism"),
            ("E66.9", "Obesity")
        ],
        # Gastrointestinal conditions
        "gastrointestinal": [
            ("K21.9", "Gastro-esophageal reflux disease"),
            ("K29.70", "Gastritis"),
            ("K58.9", "Irritable bowel syndrome"),
            ("K52.9", "Noninfective gastroenteritis and colitis")
        ],
        # Mental health conditions
        "mental_health": [
            ("F41.9", "Anxiety disorder"),
            ("F32.9", "Major depressive disorder"),
            ("F41.1", "Generalized anxiety disorder"),
            ("F43.0", "Acute stress reaction")
        ],
        # Genitourinary conditions (more common in females)
        "genitourinary_female": [
            ("N39.0", "Urinary tract infection"),
            ("N95.1", "Menopausal state"),
            ("N93.9", "Abnormal uterine and vaginal bleeding"),
            ("N76.0", "Acute vaginitis")
        ],
        # Genitourinary conditions (more common in males)
        "genitourinary_male": [
            ("N40.0", "Benign prostatic hyperplasia"),
            ("N41.0", "Acute prostatitis"),
            ("N13.5", "Kidney stone"),
            ("N45.1", "Epididymitis")
        ],
        # Common in all demographics
        "general": [
            ("R51", "Headache"),
            ("Z00.00", "General adult medical examination"),
            ("R42", "Dizziness and giddiness"),
            ("R53.83", "Fatigue")
        ]
    }
    
    # Determine relevant categories based on procedure code
    if procedure.startswith("99"):  # Office visits
        # For general visits, use demographics to determine likely conditions
        if age > 60:
            # Older patients more likely to have chronic conditions
            categories = ["cardiovascular", "endocrine", "musculoskeletal"]
            # Add gender-specific conditions
            if gender == "male":
                categories.append("genitourinary_male")
            elif gender == "female":
                categories.append("genitourinary_female")
        elif age > 40:
            # Middle-aged adults
            categories = ["endocrine", "musculoskeletal", "gastrointestinal", "mental_health"]
        else:
            # Younger adults
            categories = ["respiratory", "mental_health", "general"]
    elif procedure in ["93000"]:  # ECG
        # Heart-related procedures indicate cardiovascular diagnosis
        categories = ["cardiovascular"]
    elif procedure in ["71045"]:  # Chest X-ray
        # X-rays often for respiratory issues
        categories = ["respiratory"]
    elif procedure in ["85025", "80053"]:  # Blood tests
        # Lab work often for chronic disease management
        categories = ["endocrine", "cardiovascular"]
    elif procedure in ["97110"]:  # Physical therapy
        # Therapy for musculoskeletal issues
        categories = ["musculoskeletal"]
    else:
        # Default to general categories plus gender-specific
        categories = ["general", "respiratory", "gastrointestinal"]
    
    # Select a random category from the relevant ones, then a diagnosis from that category
    selected_category = random.choice(categories)
    selected_diagnosis = random.choice(diagnosis_codes[selected_category])
    
    # Return just the code (not the description)
    return selected_diagnosis[0]

# 2. Custom generator for CPT procedure codes
def cpt_generator(row, col_name):
    # Common CPT codes for demonstration
    common_codes = [
        "99213",  # Office visit, established patient (15 min)
        "99214",  # Office visit, established patient (25 min)
        "99203",  # Office visit, new patient (30 min)
        "90471",  # Immunization administration
        "82607",  # Vitamin B-12 blood test
        "80053",  # Comprehensive metabolic panel
        "85025",  # Complete blood count (CBC)
        "93000",  # Electrocardiogram (ECG)
        "71045",  # X-ray, chest, single view
        "97110"   # Therapeutic exercises (15 min)
    ]
    return random.choice(common_codes)

# 3. Custom generator for claim status
def claim_status_generator(row, col_name):
    statuses = ["Pending", "Approved", "Denied", "Under Review", "Appealed"]
    weights = [0.2, 0.6, 0.1, 0.05, 0.05]  # Weighted distribution
    return random.choices(statuses, weights=weights)[0]

# 4. Custom generator for realistic claim amounts
def billed_amount_generator(row, col_name):
    # Different procedures have different typical cost ranges
    cpt = row["procedure_code"]
    
    # Office visits typically $100-300
    if cpt.startswith("992"):
        return round(random.uniform(100, 300), 2)
    # Diagnostic tests typically $200-800
    elif cpt in ["82607", "80053", "85025", "93000"]:
        return round(random.uniform(200, 800), 2)
    # Imaging typically $500-1500
    elif cpt in ["71045"]:
        return round(random.uniform(500, 1500), 2)
    # Treatments typically $100-500
    else:
        return round(random.uniform(100, 500), 2)

# 5. Custom generator for allowed amounts (always less than billed)
def allowed_amount_generator(row, col_name):
    billed = row["billed_amount"]
    # Allowed amount is typically 60-90% of billed amount
    return round(billed * random.uniform(0.6, 0.9), 2)

# Register the custom generators
generator.register_generator("text", icd10_generator, column_name="primary_diagnosis")
generator.register_generator("text", cpt_generator, column_name="procedure_code")
generator.register_generator("text", claim_status_generator, column_name="status")
generator.register_generator("float", billed_amount_generator, column_name="billed_amount")
generator.register_generator("float", allowed_amount_generator, column_name="allowed_amount")

# Register a foreign key generator for patient_id
def patient_id_generator(row, col_name):
    return random.choice(patients_df["id"].tolist())

generator.register_generator("foreign_key", patient_id_generator, column_name="patient_id")

# Generate claims with the custom generators
claims_df = generator.generate_data(
    schema=Claim,
    prompt="Generate realistic health insurance claims data.",
    sample_size=200,
    output_path="claims.csv"
)

print(f"Generated {len(patients_df)} patient records and {len(claims_df)} claim records")
print(f"Claims summary: {claims_df['status'].value_counts()}")
```

This example demonstrates several powerful features of custom generators:

1. **Domain-Specific Codes**: Generate actual ICD-10 and CPT codes that conform to real-world patterns.

2. **Interdependent Values**: The `allowed_amount` generator depends on the value of `billed_amount`, showing how generators can reference other fields.

3. **Weighted Distributions**: The claim status generator uses weighted probabilities to create a realistic distribution (most claims approved, few denied).

4. **Conditional Logic**: The billed amount generator produces different ranges based on the procedure code, creating more realistic correlations.

5. **Referential Integrity**: The patient_id generator ensures each claim references a valid patient record.

Custom generators can significantly enhance the quality and realism of your synthetic data, especially for specialized domains with complex rules, codes, and relationships.

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

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details.
