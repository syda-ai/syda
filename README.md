
# Synthetic Data Generation Service -- WORK IN PROGRESS

A Python-based open-source service for generating synthetic data while preserving data utility.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
  - [Structured Data Generation](#structured-data-generation)
  - [SQLAlchemy Model Integration](#sqlalchemy-model-integration)
  - [Handling Foreign Key Relationships](#handling-foreign-key-relationships)
  - [Automatic Management of Multiple Related Models](#automatic-management-of-multiple-related-models)
  - [Complete CRM Example](#complete-crm-example)
  - [Metadata Enhancement Benefits](#metadata-enhancement-benefits)
  - [Output Options](#output-options)
- [Optional REST API Service](#optional-rest-api-service)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation-1)
  - [Running the API Service](#running-the-api-service)
  - [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Features

### Core Features

- **Synthetic Data Generation**:
  - Statistical data generation
  - Pattern-based generation
  - Data distribution preservation
  - Synthetic data from various sources

### Optional Features

- **REST API Service**:
  - Generate synthetic data via API
  - Support for CSV file uploads
  - Support for unstructured files (images, PDFs, documents)
  - JSON and CSV output formats

- **Project Management**:
  - Create and manage projects
  - Unique project names
  - Project-based transaction tracking
  - Project descriptions

- **Database Integration**:
  - PostgreSQL backend
  - Transaction logging
  - Audit trail for all operations

## Core Module Usage

### Installation

Install the package using pip:

```bash
pip install syda
```

### Basic Usage

#### Synthetic Data Generation

##### Basic Usage

You can generate synthetic data and write it directly to a CSV file using the `output_path` argument:

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

output_path = 'synthetic_output.csv'
generated_file = generator.generate_data(
    schema=schema,
    prompt=prompt,
    sample_size=15,
    output_path=output_path
)
print(f"Synthetic data written to: {generated_file}")
```

##### SQLAlchemy Model Integration with SQLAlchemy, Smart Metadata Extraction & Foreign Key Constraints

You can use SQLAlchemy model classes directly as schema input with enhanced metadata extraction that leverages model docstrings and column comments:

```python
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship
import random
import pandas as pd
from syda.structured import SyntheticDataGenerator

Base = declarative_base()

# Define models with rich docstrings and metadata
class Department(Base):
    """Organizational department that groups employees by business function.
    Departments are the primary organizational units within the company structure.
    """
    __tablename__ = 'departments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True, 
                 comment="Official department name (e.g. Engineering, Marketing)")
    location = Column(String(100), comment="Primary office location of this department")
    budget = Column(Float, comment="Annual budget allocation in USD")
    
    # One-to-many: one department has many employees
    employees = relationship("Employee", back_populates="department")


class Employee(Base):
    """Company employee with their personal and professional details.
    Each employee belongs to a specific department and has associated job information.
    """
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    first_name = Column(String(50), nullable=False, 
                       comment="Employee's given name")
    last_name = Column(String(50), nullable=False, 
                      comment="Employee's family name")
    email = Column(String(100), nullable=False, unique=True, 
                  comment="Business email address for communication")
    department_id = Column(Integer, ForeignKey('departments.id'),
                          comment="Department where employee works")
    role = Column(String(100), comment="Job title or position within the company")
    salary = Column(Float, comment="Annual salary in USD")
    
    # Many-to-one: many employees belong to one department
    department = relationship("Department", back_populates="employees")

# Step 1: Generate departments first
generator = SyntheticDataGenerator()
    
# The system automatically extracts docstrings and column metadata
# so the prompt can be much simpler
departments_df = generator.generate_data(
    schema=Department,
    prompt="Generate realistic department data for a technology company.",
    sample_size=5,
    output_path='departments.csv'
)

# Step 2: Create custom generators for specific columns
departments_df = pd.read_csv('departments.csv')

# Register a column-specific generator for department_id
def department_id_fk_generator(row, col_name):
    # Sample from the existing department IDs
    return random.choice(departments_df['id'].tolist())

# Register the custom generator specifically for the department_id column
generator.register_generator('foreign_key', department_id_fk_generator, column_name='department_id')

# Step 3: Generate employee data with valid department_id references
# The system uses model docstrings and column metadata to inform the generation
employees_df = generator.generate_data(
    schema=Employee,
    prompt="Generate realistic employee data for a technology company.",
    sample_size=10,
    output_path='employees.csv'
)

# Verify referential integrity
valid_dept_ids = set(departments_df['id'].tolist())
employee_dept_ids = set(employees_df['department_id'].tolist())

print("Verifying referential integrity...")
if employee_dept_ids.issubset(valid_dept_ids):
    print("✅ All employee department_id values reference valid departments")
else:
    invalid_ids = employee_dept_ids - valid_dept_ids
    print(f"❌ Found {len(invalid_ids)} invalid department_id references: {invalid_ids}")
```

##### Handling Foreign Key Relationships

The library provides robust support for handling foreign key relationships with referential integrity:

1. **Automatic Foreign Key Detection**: Foreign keys are automatically detected from your SQLAlchemy models and assigned the type `'foreign_key'`.

2. **Column-Specific Foreign Key Generators**: Register different generators for each foreign key column when dealing with multiple relationships:

```python
# Different generators for different foreign key columns
generator.register_generator('foreign_key', department_id_generator, column_name='department_id')
generator.register_generator('foreign_key', manager_id_generator, column_name='manager_id')
```

3. **Multi-Step Generation Process**: For related tables, generate parent records first, then use their IDs when generating child records:

```python
# Generate departments first
departments_df = generator.generate_data(schema=Department, ...)

# Then generate employees with department references
employees_df = generator.generate_data(schema=Employee, ...)

# Finally generate projects with both department and employee references
projects_df = generator.generate_data(schema=Project, ...)
```

4. **Referential Integrity Preservation**: The foreign key generator samples from actual existing IDs in the parent table, ensuring all references are valid.

5. **Metadata-Enhanced Foreign Keys**: Column comments on foreign key fields are preserved and included in the prompt, helping the LLM understand the relationship context.

##### Automatic Management of Multiple Related Models

The library also supports the automatic management of multiple related SQLAlchemy models using the `generate_related_data` method, which simplifies the process even further:

```python
# Create a generator instance
generator = SyntheticDataGenerator()

# Define custom prompts and sample sizes (optional)
prompts = {
    "Customer": "Generate diverse customer organizations for a B2B SaaS company.",
    "Product": "Generate cloud software products and services."
}

sample_sizes = {
    "Customer": 10,    # 10 customer companies
    "Contact": 25,    # 25 contact persons (distributed across customers)
    "Product": 15,    # 15 different products
    "Order": 30,      # 30 orders total
    "OrderItem": 60,  # 60 line items across all orders
}

# Define custom generators for specific columns in specific models (optional)
custom_generators = {
    "Customer": {
        # Generate one of three predefined statuses
        "status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"]),
    },
    "Product": {
        # Generate prices between $50 and $5000 with appropriate precision
        "price": lambda row, col: round(random.uniform(50, 5000), 2)
    }
}

# Generate all related data in one call with automatic dependency resolution
results = generator.generate_related_data(
    models=[Customer, Contact, Product, Order, OrderItem],
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir="output_data",  # Optional: save each model to CSV
    custom_generators=custom_generators  # Optional: use custom generators for specific columns
)

# Access the generated data as DataFrames
customers_df = results["Customer"]
orders_df = results["Order"]
```

This method:

1. **Automatically Analyzes Dependencies**: Builds a dependency graph of your models based on foreign key relationships.

2. **Determines Optimal Generation Order**: Uses topological sorting to ensure parent tables are always generated before their children.

3. **Manages Foreign Keys**: Automatically sets up foreign key generators that sample from parent table IDs.

4. **Supports Custom Prompts and Sizes**: Allows you to specify different prompts and sample sizes for each model.

5. **Supports Model-Specific Custom Generators**: Lets you define custom generators for specific columns in specific models, giving you precise control over data generation.

6. **Preserves Referential Integrity**: Ensures all generated data maintains proper relationships between tables.

Check out the complete example in `examples/test_auto_related_models.py` which shows a comprehensive CRM system with five interrelated models.

#### Complete CRM Example

Here's a complete example showing how to use `generate_related_data` with a CRM system that has multiple interrelated models:

```python
#!/usr/bin/env python

import sys
import os
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Date, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship
import datetime

# Import the synthetic data generator
from syda.structured import SyntheticDataGenerator

# Create a Base for our models
Base = declarative_base()

# Define a comprehensive set of related models for a CRM system
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


def main():
    """Demonstrate automatic generation of related data with proper foreign key handling."""
    
    # Create a generator instance
    generator = SyntheticDataGenerator()
    
    # Define output directory
    output_dir = "crm_data"
    
    # Define custom prompts for each model (optional)
    prompts = {
        "Customer": """
        Generate diverse customer organizations for a B2B SaaS company.
        Include a mix of industries like technology, healthcare, finance, etc.
        """,
        
        "Product": """
        Generate products for a cloud software company.
        Products should include various software services, support packages, and consulting.
        """,
        
        "Order": """
        Generate realistic orders with appropriate dates and statuses.
        """,
    }
    
    # Define sample sizes for each model (optional)
    sample_sizes = {
        "Customer": 10,        # Base entities
        "Contact": 25,         # ~2-3 contacts per customer
        "Product": 15,         # Products catalog
        "Order": 30,           # ~3 orders per customer
        "OrderItem": 60,       # ~2 items per order
    }
    
    # Generate data for all models with automatic dependency resolution
    results = generator.generate_related_data(
        models=[Customer, Contact, Product, Order, OrderItem],
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=output_dir
    )
    
    # Verify referential integrity
    print("\n🔍 Verifying referential integrity:")
    
    # Check Contacts → Customers
    contact_customer_ids = set(results["Contact"]["customer_id"].tolist())
    valid_customer_ids = set(results["Customer"]["id"].tolist())
    if contact_customer_ids.issubset(valid_customer_ids):
        print("  ✅ All Contact.customer_id values reference valid Customers")
    
    # Check OrderItems → Products
    order_item_product_ids = set(results["OrderItem"]["product_id"].tolist())
    valid_product_ids = set(results["Product"]["id"].tolist())
    if order_item_product_ids.issubset(valid_product_ids):
        print("  ✅ All OrderItem.product_id values reference valid Products")
```

When you run this example, the system will:

1. Automatically determine the order to generate data (Customer, Product, Contact, Order, OrderItem)
2. Generate data for each model with proper foreign key relationships
3. Save CSV files to the output directory
4. Verify that all foreign key references are valid

The rich model documentation and column metadata help the LLM generate realistic, domain-appropriate data, and the automatic dependency resolution ensures all foreign keys are valid.


##### Metadata Enhancement Benefits

This approach provides several powerful benefits:

1. **Richer Context for LLMs**: The system extracts docstrings, comments, constraints, and other metadata from your SQLAlchemy models, providing rich context that helps the LLM generate more accurate and domain-appropriate synthetic data.

2. **Simpler Prompts**: You can write much shorter prompts because the system automatically includes essential information from your model definitions.

3. **Constraint-Aware Generation**: The LLM becomes aware of constraints like `nullable=False`, `unique=True`, and field lengths, leading to data that better respects your schema rules.

4. **Column-Specific Generators**: You can register generators for specific columns (not just data types), allowing precise control over how each field is populated.

5. **Automatic Docstring Utilization**: Class and field docstrings are automatically incorporated into the generation process, ensuring the LLM understands the business context of your models.

Under the hood, the system constructs a detailed prompt that includes all relevant metadata:

```
Model Description: Company employee with their personal and professional details...

Generate 10 records JSON objects with these fields:
- id: number (primary_key)
- first_name: text (not_null) - Employee's given name
- last_name: text (not_null) - Employee's family name
- email: text (unique, not_null) - Business email address for communication
- department_id: foreign_key - Department where employee works
...
```

This allows for more accurate and contextually appropriate synthetic data generation while requiring less manual specification in your code.

##### Output Options

- If `output_path` is provided (must end with `.csv` or `.json`), the file will be written and the method returns the file path.
- If not, a pandas DataFrame is returned.

See `examples/` directory for complete examples, including:
- Basic schema-based data generation
- SQLAlchemy model integration
- Foreign key relationship handling

## Optional REST API Service

### Prerequisites
- Python 3.8+
- PostgreSQL database
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd syda-service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Database Setup

1. Create a PostgreSQL database for the application.

2. Configure the database connection by creating a `.env` file in the root directory with the following variables:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/your_database_name
   ```

3. Run Alembic migrations to set up the database tables:
   ```bash
   # Navigate to the service directory
   cd service
   
   # Create initial migration (only needed once)
   alembic revision --autogenerate -m "Initial migration"
   
   # Apply migrations
   alembic upgrade head
   ```

### Running the API Service

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### Project Management
```
POST /projects
```
Create a new project
Example request:
```json
{
    "name": "my_project",
    "description": "My synthetic data project"
}
```

```
GET /projects
```
Get all projects

```
GET /projects/{project_name}/transactions
```
Get transactions for a specific project

#### Data Operations (Project-based)

Example request:
```json
{
    "project_name": "my_project",
    "data": {
        "email": ["test@example.com"],
        "phone": ["123-456-7890"]
    }
}
```

```
POST /generate
```
Generate synthetic data for a project

```
POST /generate/test-data
```
Generate test data for a project



```
POST /upload/generate
```
Upload and generate synthetic data for a project

```
POST /upload/unstructured
```
Process unstructured file (image, PDF, Word, Excel, text)

```
POST /upload/unstructured/generate
```
Generate synthetic data from unstructured file (Excel only)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
See the [LICENSE](LICENSE) file for the full license text.
