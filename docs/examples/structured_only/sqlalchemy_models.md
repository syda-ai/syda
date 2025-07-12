# SQLAlchemy Models

> Source code: [examples/structured_only/example_sqlalchemy_models.py](https://github.com/syda-ai/syda/blob/main/examples/structured_only/test_sqlalchemy_models.py)

This example demonstrates how to use SQLAlchemy models for synthetic data generation with SYDA.

## Overview

SQLAlchemy is a popular ORM (Object Relational Mapper) for Python. SYDA can directly use your SQLAlchemy models to generate synthetic data that respects your database schema, including foreign key relationships and constraints.

## Model Definition

SQLAlchemy models define your data structures using Python classes. SYDA automatically analyzes these models to understand:
- Table structures and field types
- Primary keys
- Foreign key relationships
- Column constraints
- Documentation comments

Here are examples of SQLAlchemy models for a CRM system:

### Customer Model

```python
class Customer(Base):
    """Organization or individual client in the CRM system.
    Represents a business entity that can place orders and have contacts.
    """
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True, 
                 comment="Customer organization name")
    industry = Column(String(50), comment="Customer's primary industry")
    website = Column(String(100), comment="Customer's website URL")
    status = Column(String(20), comment="Active, Inactive, Prospect")
    created_at = Column(Date, default=datetime.date.today,
                       comment="Date when customer was added to CRM")
    
    # Relationships
    contacts = relationship("Contact", back_populates="customer")
    orders = relationship("Order", back_populates="customer")
```

### Contact Model

```python
class Contact(Base):
    """Individual person associated with a customer organization.
    Contacts are the individuals we communicate with at the customer.
    """
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
    phone = Column(String(20), comment="Contact's phone number")
    position = Column(String(100), comment="Job title or position")
    is_primary = Column(Boolean, default=False, 
                       comment="Whether this is the primary contact")
    
    # Relationships
    customer = relationship("Customer", back_populates="contacts")
```

### Product Model

```python
class Product(Base):
    """Product or service offered by the company.
    Products can be ordered by customers.
    """
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True,
                 comment="Product name")
    category = Column(String(50), comment="Product category")
    price = Column(Float, nullable=False, comment="Product price in USD")
    description = Column(Text, comment="Detailed product description")
    
    # Relationships
    order_items = relationship("OrderItem", back_populates="product")
```

### Order Model

```python
class Order(Base):
    """Customer order for products or services.
    Orders contain order items for specific products.
    """
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False,
                        comment="Customer who placed the order")
    order_date = Column(Date, nullable=False, 
                       comment="Date when order was placed")
    status = Column(String(20), comment="New, Processing, Shipped, Delivered, Cancelled")
    total_amount = Column(Float, comment="Total order amount in USD")
    
    # Relationships
    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")
```

### OrderItem Model

```python
class OrderItem(Base):
    """Individual item within an order.
    Each order item represents a specific product in a specific quantity.
    """
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False,
                     comment="Order this item belongs to")
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False,
                       comment="Product being ordered")
    quantity = Column(Integer, nullable=False, 
                     comment="Quantity of product ordered")
    unit_price = Column(Float, nullable=False, 
                       comment="Price per unit at time of order")
    
    # Relationships
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")
```

## Foreign Key Handling

SQLAlchemy models define foreign keys explicitly through ForeignKey column definitions, which SYDA uses for detection:

```python
customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
```

SYDA automatically analyzes these foreign key definitions to:
- Determine the correct generation order (parent tables first)
- Ensure referential integrity in the generated data
- Handle one-to-many and many-to-one relationships

Note: While SQLAlchemy models also define relationships using the `relationship()` function, SYDA specifically looks for the explicit `ForeignKey()` definitions to detect dependencies between tables.

## Code Example

Here's how to use SQLAlchemy models with the SyntheticDataGenerator:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig
import os
import random
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Date, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship

# Create a Base for our models
Base = declarative_base()

# Define SQLAlchemy models (Customer, Contact, Product, etc.)
# ... model definitions as shown above ...

# Create a generator instance
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=4000,
)
generator = SyntheticDataGenerator(model_config=model_config)

# Define output directory
output_dir = "output/example_sqlalchemy_models/crm_data"

# Define custom prompts
prompts = {
    "customers": """
    Generate diverse customer organizations for a B2B SaaS company.
    Include a mix of industries like technology, healthcare, finance, etc.
    """,
    
    "products": """
    Generate products for a cloud software company.
    Products should include various software services, support packages, and consulting.
    """,
    
    "orders": """
    Generate realistic orders with appropriate dates and statuses.
    """,
}

# Define sample sizes
sample_sizes = {
    "customers": 10,        # Base entities
    "contacts": 25,         # ~2-3 contacts per customer
    "products": 15,         # Products catalog
    "orders": 30,           # ~3 orders per customer
    "order_items": 60,       # ~2 items per order
}

# Define custom generators for specific model columns
# NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
# based on column names, SQLAlchemy comments, and field types.
custom_generators = {
    "customers": {
        # Ensure a specific distribution of customer statuses for business reporting
        "status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"]),
    },
    "products": {
        # Control price ranges more precisely than the AI might
        "price": lambda row, col: round(random.uniform(50, 5000), 2),
        # Ensure product categories match your specific business domains
        "category": lambda row, col: random.choice([
            "Cloud Infrastructure", "Business Intelligence", "Security Services",
            "Data Analytics", "Custom Development", "Support Package", "API Services"
        ])
    },
    "order_items": {
        # Example of a simple numeric distribution
        "quantity": lambda row, col: random.randint(1, 10),
    },
}

# Generate data for all SQLAlchemy models with automatic dependency resolution
results = generator.generate_for_sqlalchemy_models(
    sqlalchemy_models=[Customer, Contact, Product, Order, OrderItem],
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir=output_dir,
    custom_generators=custom_generators
)
```

## Key Features

1. **Native ORM Integration**: Works directly with your SQLAlchemy models
2. **Automatic Relationship Detection**: Analyzes foreign keys and relationships
3. **Comment-Based Guidance**: Uses SQLAlchemy column comments for better AI prompting
4. **Type Awareness**: Respects column types and constraints
5. **Dependency Resolution**: Handles the correct order for generating related data

## Best Practices

1. **Add Comments**: Use SQLAlchemy comments to guide data generation
   ```python
   name = Column(String(100), comment="Customer organization name")
   ```

2. **Define Relationships**: Always define proper relationships between models
   ```python
   contacts = relationship("Contact", back_populates="customer")
   ```

3. **Use Constraints**: Add constraints like nullable=False and unique=True
   ```python
   email = Column(String(100), nullable=False, unique=True)
   ```

4. **Custom Generators**: Use custom generators for fields needing specific distributions
   ```python
   "status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"])
   ```

5. **Model Documentation**: Add docstrings to describe your models
   ```python
   """Organization or individual client in the CRM system."""
   ```

## Sample Outputs

You can view sample outputs generated using these SQLAlchemy models here:

> [Example SQLAlchemy Model Outputs](https://github.com/syda-ai/syda/tree/main/examples/structured_only/output/test_sqlalchemy_models/crm_data)
