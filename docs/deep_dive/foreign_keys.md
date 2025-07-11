# Foreign Key Handling

One of SYDA's most powerful features is its ability to maintain referential integrity across multiple related tables. This document explains in detail how foreign key relationships are defined, detected, and handled during data generation.

## Foreign Key Definition Methods

SYDA supports three different ways to define foreign key relationships:

### 1. Special `__foreign_keys__` Section

You can explicitly define foreign keys using a special `__foreign_keys__` section in your schema:

```python
schemas = {
    'Order': {
        'id': {'type': 'integer', 'primary_key': True},
        'customer_id': {'type': 'integer'},
        'order_date': {'type': 'date'},
        # Special section defining foreign keys
        '__foreign_keys__': {
            'customer_id': {'table': 'Customer', 'column': 'id'}
        }
    }
}
```

This approach is particularly useful when:
- You're using dictionary-based schemas
- You need to define multiple foreign keys
- The field name doesn't follow naming conventions

### 2. Field-Level `references` Property

You can define foreign keys directly in field definitions using the `references` property:

```python
schemas = {
    'Order': {
        'id': {'type': 'integer', 'primary_key': True},
        'customer_id': {
            'type': 'integer',
            'references': {'table': 'Customer', 'column': 'id'}
        },
        'order_date': {'type': 'date'}
    }
}
```

This approach keeps the foreign key definition close to the field it applies to, making the schema more readable.

### 3. SQLAlchemy `ForeignKey` Definitions

When using SQLAlchemy models, foreign keys are automatically detected from the `ForeignKey` definitions:

```python
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    order_date = Column(Date)
```

SYDA will automatically extract these relationships during schema analysis.

## Dependency Resolution

Once foreign keys are defined, SYDA automatically determines the correct order for generating data:

1. **Dependency Graph**: SYDA builds a directed graph of dependencies between tables
2. **Topological Sort**: It performs a topological sort to determine the generation order
3. **Execution Order**: Tables are generated in an order that ensures all parent tables exist first

For example, with these tables:
- Customer (no dependencies)
- Product (no dependencies)
- Order (depends on Customer)
- OrderItem (depends on Order and Product)

SYDA would generate them in this order:
1. Customer and Product (can be generated in parallel)
2. Order (after Customer is available)
3. OrderItem (after both Order and Product are available)

## Self-Referential Relationships

SYDA can handle self-referential relationships, such as employee-manager hierarchies:

```python
class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    manager_id = Column(Integer, ForeignKey('employees.id'), nullable=True)
```

For self-referential relationships:
1. The table is generated in multiple passes
2. First pass creates records with null foreign keys
3. Subsequent passes update records to establish the relationships

## Foreign Key Value Assignment

When generating data with foreign keys, SYDA ensures that each foreign key references a valid primary key in the parent table:

1. **Parent Table Access**: SYDA maintains access to all previously generated tables
2. **Random Selection**: By default, it randomly selects a valid foreign key value
3. **Distribution Control**: You can control the distribution pattern using custom generators

Example of a custom generator controlling foreign key distribution:

```python
def distribute_manager_assignments(table_name, column_name, row_data, dependencies=None):
    """
    Assign managers with a specific distribution pattern.
    20% of employees have no manager (executives)
    80% of employees are distributed evenly among managers
    """
    if random.random() < 0.2:
        return None  # Executive with no manager
    
    # Get all employees who could be managers
    if not dependencies or 'employees' not in dependencies:
        return None
    
    # Find potential managers (those already generated)
    potential_managers = [e['id'] for e in dependencies['employees']]
    
    if not potential_managers:
        return None
        
    # Select a manager randomly
    return random.choice(potential_managers)
```

## Best Practices for Foreign Key Handling

1. **Be Explicit**: Whenever possible, explicitly define foreign key relationships
2. **Consistent Naming**: Use consistent naming patterns (e.g., `table_id`) for foreign keys
3. **Handle Nullable Keys**: Specify whether foreign keys can be null
4. **Consider Distribution**: Use custom generators to control how foreign keys are distributed
5. **Test Relationships**: Verify that generated data maintains proper referential integrity
6. **Document Dependencies**: Add comments or documentation about table dependencies

## Common Foreign Key Patterns

### One-to-Many Relationships

```python
# One customer has many orders
class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
```

### Many-to-Many Relationships

```python
# Many products in many categories through a junction table
class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))

class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))

class ProductCategory(Base):
    __tablename__ = 'product_categories'
    product_id = Column(Integer, ForeignKey('products.id'), primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.id'), primary_key=True)
```

### Self-Referential Hierarchies

```python
# Hierarchical organization structure
class Department(Base):
    __tablename__ = 'departments'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    parent_dept_id = Column(Integer, ForeignKey('departments.id'), nullable=True)
```
