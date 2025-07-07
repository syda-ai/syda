# SYDA Schema Reference

This document defines the schema format for the SYDA (Synthetic Data) library. It describes the supported field types, constraints, and special schema sections.

## Basic Schema Structure

A schema in SYDA is defined as a JSON or YAML object with field names as keys and field types as values. The simplest form is:

```yaml
field1: text
field2: number
field3: date
```

## Field Types

The following field types are supported:

| Type | Description | Example |
|------|-------------|---------|
| `text` | Generic text field | `"name": "text"` |
| `string` | Same as text | `"title": "string"` |
| `number` | Numeric field (integer or decimal) | `"id": "number"` |
| `integer` | Integer value (alias: `int`) | `"age": "integer"` |
| `float` | Floating-point number | `"price": "float"` |
| `date` | Date value | `"birth_date": "date"` |
| `datetime` | Date and time value | `"created_at": "datetime"` |
| `boolean` | True/false value (alias: `bool`) | `"is_active": "boolean"` |
| `email` | Email address | `"contact": "email"` |
| `phone` | Phone number | `"telephone": "phone"` |
| `address` | Physical address | `"location": "address"` |
| `url` | URL/website address | `"website": "url"` |
| `array` | List of items | `"items": "array"` |
| `foreign_key` | Reference to another table | `"user_id": "foreign_key"` |

## Enhanced Field Definitions

Fields can be defined with additional properties by using an object instead of a string:

```yaml
name:
  type: text
  description: Full name of the person
  constraints:
    max_length: 50
```

### Supported Field Properties

| Property | Description | Example |
|----------|-------------|---------|
| `type` | The field data type (required) | `"type": "text"` |
| `description` | Human-readable description | `"description": "User's email"` |
| `length` | Fixed length for the field | `"length": 10` |
| `min_length` | Minimum length | `"min_length": 5` |
| `max_length` | Maximum length | `"max_length": 100` |
| `min` | Minimum value (for numeric types) | `"min": 0` |
| `max` | Maximum value (for numeric types) | `"max": 1000` |
| `decimals` | Number of decimal places (for float) | `"decimals": 2` |
| `format` | Format string (for dates, etc.) | `"format": "YYYY-MM-DD"` |
| `enum` | List of possible values | `"enum": ["active", "inactive"]` |
| `pattern` | Regex pattern | `"pattern": "^[A-Z][a-z]+$"` |
| `references` | Foreign key reference | `"references": {"schema": "User", "field": "id"}` |

> **Note**: Field properties are validated during schema validation. For example, using an invalid field type will cause validation to fail with a detailed error message.

### Field Constraints

Constraints can be specified directly in the field definition or in a separate `constraints` object:

```yaml
email:
  type: email
  constraints:
    unique: true
    not_null: true
```

| Constraint | Description | Example |
|------------|-------------|---------|
| `not_null` | Field cannot be null/empty | `"not_null": true` |
| `unique` | Values must be unique | `"unique": true` |
| `primary_key` | Field is a primary key | `"primary_key": true` |
| `format` | Format for specialized types | `"format": "email"` |
| `pattern` | Regex pattern | `"pattern": "^[A-Z][a-z]+$"` |
| `min` | Minimum value (numeric) | `"min": 0` |
| `max` | Maximum value (numeric) | `"max": 1000` |
| `min_length` | Minimum string length | `"min_length": 5` |
| `max_length` | Maximum string length | `"max_length": 50` |
| `length` | Exact string length | `"length": 10` |

> **Note**: Constraints are validated during schema validation. For example, if both `length` and `max_length` are specified for a string field, validation will fail with an appropriate error message.

## Special Schema Sections

Special sections in the schema are prefixed with double underscores. These special sections are validated during schema validation:

### `__description__`

It is used to identify the table description for the schema.

```yaml
__description__: Customer information for e-commerce site
```
### `__table_description__`

It can also be used to identify the table description for the schema.

```yaml
__table_description__: Customer information for e-commerce site
```

### `__foreign_keys__`

Defines foreign key relationships:

```yaml
__foreign_keys__:
  user_id: [User, id]
  product_id: [Product, id]
```

### `__depends_on__`

Specifies schema dependencies for generation order:

```yaml
__depends_on__: [Product, Customer]
```

This ensures that Product and Customer data are generated before the current schema.


## Special Template-Related Fields

For schemas that generate unstructured document outputs:

### `__template__`

It can be set to `true` or a string value to enable template generation.

```yaml
__template__: true
```

### `__template_source__`

It is used to specify the path to the template file.

```yaml
__template_source__: /path/to/template.html
```

### `__input_file_type__`

It is used to specify the input file type.

```yaml
__input_file_type__: html
```

### `__output_file_type__`

It is used to specify the output file type.

```yaml
__output_file_type__: pdf
```

These fields enable document generation from templates with the synthetic data.

> **Important**: When `__template__` is set to `true`, the `__template_source__` field is required. Schema validation will fail if this relationship is not maintained.

## Multiple Ways to Define Foreign Keys

Foreign keys can be defined in three ways:

1. Using the `__foreign_keys__` special section (recommended):
   ```yaml
   __foreign_keys__:
     user_id: [User, id]
   user_id: foreign_key
   ```

2. Using the field definition with `references`:
   ```yaml
   user_id:
     type: foreign_key
     references:
       schema: User
       field: id
   ```

3. Using naming conventions (field name must end with `_id`):
   ```yaml
   user_id: number
   ```

## Array Fields

Array fields can be defined to contain lists of items:

```yaml
items:
  type: array
  description: "List of purchased items with product details"
```

Arrays are typically populated using custom generators in the application code.


## Structured Data Example

```yaml
__table_name__: Order
__table_description__: Orders for an e-commerce system
__depends_on__: [Customer, Product]
__foreign_keys__:
  customer_id: [Customer, id]

id:
  type: integer
  description: Order ID
  constraints:
    primary_key: true
    not_null: true

customer_id:
  type: integer
  description: Customer reference
  constraints:
    not_null: true

order_date:
  type: date
  constraints:
    format: YYYY-MM-DD
    not_null: true

total_amount:
  type: float
  description: Total order amount
  constraints:
    min: 0
    decimals: 2

status:
  type: text
  description: Order processing status
  constraints:
    enum: ["pending", "processing", "shipped", "delivered", "cancelled"]

items:
  type: array
  description: Line items in the order
```


## Template Schema Example

```yaml
__template__: true
__table_description__: Invoice document template
__name__: Invoice
__depends_on__: [Customer, Order, OrderItem]
__foreign_keys__:
  customer_id: [Customer, id]
  order_id: [Order, id]

__template_source__: /path/to/invoice_template.html
__input_file_type__: html
__output_file_type__: pdf

invoice_number:
  type: string
  pattern: 'INV-\d{6}'
  description: Unique invoice identifier

customer_id:
  type: integer
  description: Reference to customer

order_id:
  type: integer
  description: Reference to order

issue_date:
  type: date
  format: YYYY-MM-DD
  description: Date when invoice was issued

due_date:
  type: date
  format: YYYY-MM-DD
  description: Payment due date

items:
  type: array
  description: Line items from the order
```