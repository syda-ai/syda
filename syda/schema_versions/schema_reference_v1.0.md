# SYDA Schema Reference - v1.0

This document defines the schema format for the SYDA (Synthetic Data) library. It describes the supported field types, constraints, and special schema sections.

## Basic Schema Structure

A schema in SYDA is defined as a JSON or YAML object with field names as keys and field types as values. The simplest form is:

```json
{
  "field1": "text",
  "field2": "number",
  "field3": "date"
}
```

## Field Types

The following field types are supported:

| Type | Description | Example |
|------|-------------|---------|
| `text` | Generic text field | `"name": "text"` |
| `string` | Same as text | `"title": "string"` |
| `number` | Numeric field (integer or decimal) | `"id": "number"` |
| `integer` | Integer value | `"age": "integer"` |
| `float` | Floating-point number | `"price": "float"` |
| `date` | Date value | `"birth_date": "date"` |
| `datetime` | Date and time value | `"created_at": "datetime"` |
| `boolean` | True/false value | `"is_active": "boolean"` |
| `email` | Email address | `"contact": "email"` |
| `phone` | Phone number | `"telephone": "phone"` |
| `address` | Physical address | `"location": "address"` |
| `foreign_key` | Reference to another table | `"user_id": "foreign_key"` |

## Enhanced Field Definitions

Fields can be defined with additional properties by using an object instead of a string:

```json
{
  "name": {
    "type": "text",
    "length": 50,
    "description": "Full name of the person"
  }
}
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

### Field Constraints

Constraints can be specified directly in the field definition or in a separate `constraints` object:

```json
{
  "email": {
    "type": "email",
    "constraints": {
      "unique": true,
      "required": true
    }
  }
}
```

| Constraint | Description | Example |
|------------|-------------|---------|
| `required` | Field cannot be null/empty | `"required": true` |
| `unique` | Values must be unique | `"unique": true` |
| `primary_key` | Field is a primary key | `"primary_key": true` |

## Special Schema Sections

Special sections in the schema are prefixed with double underscores:

### `__table_description__`

Provides a description of the table/entity:

```json
{
  "__table_description__": "Customer information for e-commerce site",
  "id": "number",
  "name": "text"
}
```

### `__foreign_keys__`

Defines foreign key relationships:

```json
{
  "__foreign_keys__": {
    "user_id": ["User", "id"],
    "product_id": ["Product", "id"]
  },
  "user_id": "foreign_key",
  "product_id": "foreign_key"
}
```

### `__template__`

For template schemas, defines the template source and file types:

```json
{
  "__template__": {
    "source": "/path/to/template.html",
    "input_file_type": "html",
    "output_file_type": "pdf"
  }
}
```

## Multiple Ways to Define Foreign Keys

Foreign keys can be defined in three ways:

1. Using the `__foreign_keys__` special section (recommended):
   ```json
   {
     "__foreign_keys__": {
       "user_id": ["User", "id"]
     },
     "user_id": "foreign_key"
   }
   ```

2. Using the field definition with `references`:
   ```json
   {
     "user_id": {
       "type": "foreign_key",
       "references": {
         "schema": "User",
         "field": "id"
       }
     }
   }
   ```

3. Using naming conventions (field name must end with `_id`):
   ```json
   {
     "user_id": "number"
   }
   ```

## Complete Example

```json
{
  "__table_description__": "Orders for an e-commerce system",
  "__foreign_keys__": {
    "customer_id": ["Customer", "id"]
  },
  "id": {
    "type": "number",
    "description": "Order ID",
    "constraints": {
      "primary_key": true
    }
  },
  "customer_id": "foreign_key",
  "order_date": {
    "type": "date",
    "format": "YYYY-MM-DD"
  },
  "total_amount": {
    "type": "float",
    "min": 0,
    "decimals": 2,
    "description": "Total order amount"
  },
  "status": {
    "type": "text",
    "enum": ["pending", "processing", "shipped", "delivered", "cancelled"]
  }
}
```
