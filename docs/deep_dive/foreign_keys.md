# Foreign Key Handling

One of SYDA's most powerful features is its ability to maintain referential integrity across multiple related tables. This document explains in detail how foreign key relationships are defined, detected, and handled during data generation.

## Foreign Key Definition Methods

SYDA supports three different ways to define foreign key relationships:

### 1. Special `__foreign_keys__` Section

You can explicitly define foreign keys using a special `__foreign_keys__` section in your schema:

```yaml
# order.yaml
id:
  type: integer
  primary_key: true
customer_id:
  type: integer
order_date:
  type: date

__foreign_keys__:
  customer_id: [Customer, id]
    
```


### 2. Field-Level `references` Property

You can define foreign keys directly in field definitions using the `references` property:

```yaml
# orderitem.yaml
id:
  type: integer
  primary_key: true
order_id:
  type: integer
  references:
    schema: Order
    field: id
product_id:
  type: integer
  references:
    schema: Product
    field: id
quantity:
  type: integer
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

## Foreign Key Value Assignment

When generating data with foreign keys, SYDA ensures that each foreign key references a valid primary key in the parent table:

1. **Parent Table Access**: SYDA maintains access to all previously generated tables
2. **Random Selection**: By default, it randomly selects a valid foreign key value
3. **Consistent Foreign Keys**: When multiple columns in the same schema reference the same parent table, SYDA ensures they get the same parent record for consistency



## Best Practices for Foreign Key Handling

1. **Be Explicit**: Whenever possible, explicitly define foreign key relationships
2. **Consistent Naming**: Use consistent naming patterns (e.g., `table_id`) for foreign keys
3. **Handle Nullable Keys**: Specify whether foreign keys can be null
4. **Test Relationships**: Verify that generated data maintains proper referential integrity
5. **Document Dependencies**: Add comments or documentation about table dependencies

## Examples

To see foreign key relationships in action, explore the example projects included with SYDA:

1. **SQLAlchemy Examples**: Check [sqlalchemy_models](../examples/structured_and_unstructured_mixed/sqlalchemy_models.md) for examples of foreign keys with SQLAlchemy models
2. **Dictionary Schema Examples**: See [Dictionary Examples](../examples/structured_only/dict_schemas.md) for dictionary-based foreign key handling
3. **YAML/JSON Schema Examples**: The [YAML Examples](../examples/structured_only/yaml_schemas.md) and [JSON Examples](../examples/structured_only/json_schemas.md)  demonstrate foreign keys in file-based schemas
4. **Retail Example**: [Retail Example](../examples/structured_and_unstructured_mixed/yaml_schemas.md) shows foreign keys connecting multiple related tables

Each example demonstrates different aspects of foreign key handling, including relationship definition, value assignment, and referential integrity verification.
