# Structured Data Generation



The central class in SYDA is `SyntheticDataGenerator`, which provides methods for generating data from different schema formats:

```python
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the model
config = ModelConfig(
    provider="anthropic", 
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8000
)

# Initialize the generator
generator = SyntheticDataGenerator(model_config=config)
```

SYDA supports three primary methods for generating structured data.

For detailed information on supported field types, special field types, and format options, please refer to the [Schema Reference](../schema_reference/field_types.md) section.

### 1. YAML/JSON Schemas

You can load schemas from YAML or JSON files, which is useful for storing schema definitions outside your code:

```python
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a generator instance
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192
)
generator = SyntheticDataGenerator(model_config=model_config)

# Define paths to schema files
schema_dir = "schema_files/yaml"

# Dictionary mapping schema names to their YAML file paths
schemas = {
    "Product": os.path.join(schema_dir, "product.yml"),
    "Category": os.path.join(schema_dir, "category.yml")
}

# Define custom prompts for each schema (optional)
prompts = {
    "Product": "Generate diverse products for an inventory management system.",
    "Category": "Generate product categories for an inventory system."
}

# Define sample sizes for each schema
sample_sizes = {
    "Product": 20,      # Products
    "Category": 5       # Categories for products
}

# Generate data
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes=sample_sizes,
    prompts=prompts,
    output_dir="output/inventory_data"
)
```

Example YAML schema:

```yaml
# product.yml
__table_name__: Product
__description__: Retail products
__foreign_keys__:
  category_id: [Category, id]

id:
  type: integer
  primary_key: true
name:
  type: string
category_id:
  type: integer
```

```yaml
# category.yml
__table_name__: Category
__description__: Retail product categories
__foreign_keys__:
  parent_id: [Category, id]

id:
  type: integer
  primary_key: true
name:
  type: string
parent_id:
  type: integer
```

### 2. Dictionary-Based Schemas

Python dictionaries to define schemas:

```python
# Define schemas as dictionaries
schemas = {
    # Customer schema with table and column descriptions
    'Customer': {
        '__table_description__': 'Registered users of the e-commerce platform who can place orders',
        'id': {'type': 'number', 'description': 'Unique identifier for the customer'},
        'name': {'type': 'text', 'description': 'Full name of the customer'},
        'email': {'type': 'email', 'description': 'Customer email address used for communication'},
        'signup_date': {'type': 'date', 'description': 'Date when the customer registered'},
        'loyalty_tier': {'type': 'text', 'description': 'Customer loyalty program level (Bronze, Silver, Gold, Platinum)'}
    },
    
    # Order schema with table and column descriptions and foreign keys
    'Order': {
        '__table_description__': 'Customer orders for products, including order status and total amount',
        '__foreign_keys__': {
            'customer_id': ['Customer', 'id']  # Order.customer_id references Customer.id
        },
        'id': {'type': 'number', 'description': 'Unique order identifier', 'primary_key': True},
        'customer_id': {'type': 'foreign_key', 'description': 'Reference to the customer who placed the order'},
        'order_date': {'type': 'date', 'description': 'Date when the order was placed'},
        'status': {'type': 'text', 'description': 'Current status of the order (Pending, Processing, Shipped, Delivered, Cancelled)'},
        'total_amount': {'type': 'number', 'description': 'Total amount of the order in USD'},
        'shipping_address': {'type': 'text', 'description': 'Address where the order should be delivered'}
    }
}

# Create a generator instance with appropriate model config
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192  # Using higher max_tokens value for more complete responses
)
generator = SyntheticDataGenerator(model_config=model_config)

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



### 3. SQLAlchemy Models

For applications already using SQLAlchemy, SYDA can work directly with your models:

```python
from syda import SyntheticDataGenerator, ModelConfig
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

Base = declarative_base()

class Customer(Base):
    """Organization or individual client in the CRM system."""
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True, 
                 comment="Customer organization name")
    industry = Column(String(50), comment="Customer's primary industry")
    website = Column(String(100), comment="Customer's website URL")
    status = Column(String(20), comment="Active, Inactive, Prospect")
    
    # Define relationships
    contacts = relationship("Contact", back_populates="customer")
    orders = relationship("Order", back_populates="customer")

class Contact(Base):
    """Individual person associated with a customer organization."""
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False,
                        comment="Customer this contact belongs to")
    first_name = Column(String(50), nullable=False, 
                       comment="Contact's first name")
    last_name = Column(String(50), nullable=False, 
                      comment="Contact's last name")
    email = Column(String(100), nullable=False, unique=True, 
                  comment="Contact's email address")
    
    # Define relationship
    customer = relationship("Customer", back_populates="contacts")

# Create a generator instance with appropriate model config
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=4000
)
generator = SyntheticDataGenerator(model_config=model_config)

# Define custom generators for specific fields (optional)


# Generate data for all SQLAlchemy models with automatic dependency resolution
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Customer, Contact],
    prompts={
        "customers": "Generate diverse customer organizations for a B2B SaaS company.",
        "contacts": "Generate realistic contact information for business professionals."
    },
    sample_sizes={
        "customers": 10,
        "contacts": 25,
    },
    output_dir="output/crm_data"
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

The generated data is returned as a dictionary of pandas DataFrames and also can save the data to a directory, refer to [Output Options](output_options.md) for more details.

```python
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={'Customer': 10, 'Order': 25},
    output_dir="output/ecommerce"
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

```

## Examples

To see structured data generation in action, explore  


[Yaml Example](../examples/structured_only/yaml_schemas.md)

[JSON Example](../examples/structured_only/json_schemas.md) 

[Dict Example](../examples/structured_only/dict_schemas.md) 

[SQLAlchemy Example](../examples/structured_only/sqlalchemy_models.md) 




