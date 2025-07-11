# Core API

SYDA's core API provides a powerful and flexible framework for generating synthetic data across multiple related tables while maintaining referential integrity. This document provides a comprehensive overview of the key components and usage patterns.

## SyntheticDataGenerator

The central class in SYDA is `SyntheticDataGenerator`, which provides methods for generating data from different schema formats:

```python
from syda import SyntheticDataGenerator, ModelConfig

# Configure the model
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")

# Initialize the generator
generator = SyntheticDataGenerator(model_config=config)
```

## Structured Data Generation

SYDA supports three primary methods for generating structured data. For detailed information on supported field types, special field types, and format options, please refer to the [Schema Reference](../schema_reference/field_types.md) section.

### 1. Dictionary-Based Schemas

The most flexible approach uses Python dictionaries to define schemas:

```python
# Define schemas as dictionaries
schemas = {
    'Customer': {
        'id': {'type': 'integer', 'primary_key': True},
        'name': {'type': 'string'},
        'email': {'type': 'string', 'format': 'email'},
        'registration_date': {'type': 'date'}
    },
    'Order': {
        'id': {'type': 'integer', 'primary_key': True},
        'customer_id': {
            'type': 'integer',
            'references': {'table': 'Customer', 'column': 'id'}
        },
        'order_date': {'type': 'date'},
        'total_amount': {'type': 'number', 'format': 'float'}
    }
}

# Generate data
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={
        'Customer': 10,
        'Order': 25
    },
    prompts={
        'Customer': 'Generate realistic customer data for an e-commerce store',
        'Order': 'Generate order data with reasonable purchase amounts and dates'
    },
    output_dir='output/data'
)
```

### 2. YAML/JSON Schemas

You can load schemas from YAML or JSON files, which is useful for storing schema definitions outside your code:

```python
from syda import SyntheticDataGenerator, ModelConfig
import yaml

# Load schemas from YAML files
with open('schemas/customer.yaml', 'r') as f:
    customer_schema = yaml.safe_load(f)
    
with open('schemas/order.yaml', 'r') as f:
    order_schema = yaml.safe_load(f)
    
schemas = {
    'Customer': customer_schema,
    'Order': order_schema
}

# Generate data
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={'Customer': 10, 'Order': 25}
)
```

Example YAML schema:

```yaml
# customer.yaml
id:
  type: integer
  primary_key: true
name:
  type: string
email:
  type: string
  format: email
registration_date:
  type: date
```

### 3. SQLAlchemy Models

For applications already using SQLAlchemy, SYDA can work directly with your models:

```python
from syda import SyntheticDataGenerator, ModelConfig
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), comment="Customer's full name")
    email = Column(String(100), comment="Email address")
    registration_date = Column(Date, comment="When the customer registered")
    
class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    order_date = Column(Date, comment="When the order was placed")
    total_amount = Column(Float, comment="Total order amount in USD")
    
    # Define relationship
    customer = relationship("Customer")

# Generate data from SQLAlchemy models
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Customer, Order],
    sample_sizes={
        'customers': 10,
        'orders': 25
    },
    prompts={
        'customers': 'Generate realistic customer data',
        'orders': 'Generate order data with valid amounts'
    },
    output_dir='output/sqlalchemy_data'
)
```



## Multi-Model Generation

SYDA supports generating data for multiple related models in a single operation:

```python
# Define multiple related schemas
schemas = {
    'Customer': {...},
    'Product': {...},
    'Order': {...},
    'OrderItem': {...},
    'Invoice': {...}
}

# Generate data for all schemas
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={
        'Customer': 10,
        'Product': 20,
        'Order': 30,
        'OrderItem': 75,
        'Invoice': 30
    },
    prompts={...},
    output_dir='output/ecommerce'
)
```

SYDA automatically:

1. Determines the correct generation order based on dependencies

2. Ensures referential integrity between tables

3. Provides access to related records during generation

## Prompting Strategies

Effective prompts significantly improve the quality of generated data:

```python
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={'Customer': 10},
    prompts={
        'Customer': 'Generate diverse customer data for a B2B software company. Include customers from various industries, with realistic company names, and valid email domains. Make some customers enterprise level and others small businesses.'
    }
)
```

Best practices for prompts:

1. Be specific about the domain and context

2. Specify the range and distribution of values where relevant

3. Mention any constraints or patterns to follow

4. Provide examples for complex or unusual formats

## Working with Results

The generated data is returned as a dictionary of pandas DataFrames:

```python
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={'Customer': 10, 'Order': 25}
)

# Access the generated DataFrames
customer_df = results['Customer']
order_df = results['Order']

# Basic analysis
print(f"Generated {len(customer_df)} customer records")
print(f"Generated {len(order_df)} order records")

# Data exploration
print(customer_df.head())
print(order_df.describe())

# Save to different formats
customer_df.to_csv('customers_export.csv')
order_df.to_excel('orders_export.xlsx')
```

## Advanced Configuration

SYDA provides several options for advanced configuration:

```python
# Configure LLM parameters
config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.4,  # Lower temperature for more consistent results
    max_tokens=4000   # Adjust token limit for complex schemas
)

# Generate with advanced options
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts={"Customer": "Generate diverse customers for an e-commerce platform."},
    sample_sizes={"Customer": 10, "Order": 25},
    default_sample_size=5,   # Default number of records for schemas not in sample_sizes
    default_prompt="Generate synthetic data",  # Default prompt for schemas not in prompts
    output_dir="output",   # Save results to this directory
    output_format="csv",   # Format to save files in (csv or json)
    custom_generators={...}  # Use custom generator functions
)
```


