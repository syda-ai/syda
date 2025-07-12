# SYDA Schema Reference

This document defines the schema format for the SYDA (Synthetic Data) library. It describes the supported field types and constraints.

## Basic Schema Structure

A schema in SYDA is defined as a JSON or YAML object with field names as keys and field types as values. The simplest form is:

```yaml
field1:
  type: string
field2:
  type: number
field3:
  type: date
```

## Field Types

The following field types are supported:

### text

Generic text field for storing names, descriptions, and other textual content.

**Example:**
```yaml
name:
  type: text
```

### string

Same as text, used for storing character data.

**Example:**
```yaml
title:
  type: string
```

### number

Numeric field that can represent integers or decimal values.

**Example:**
```yaml
id:
  type: number
```

### integer

Integer value with no decimal places. Can also use the alias `int`.

**Example:**
```yaml
age:
  type: integer
```

### float

Floating-point number that can have decimal places.

**Example:**
```yaml
price:
  type: float
```

### date

Date value in standard format.

**Example:**
```yaml
birth_date:
  type: date
```

### datetime

Date and time value.

**Example:**
```yaml
created_at:
  type: datetime
```

### boolean

True/false value. Can also use the alias `bool`.

**Example:**
```yaml
is_active:
  type: boolean
```

### email

Email address with appropriate validation.

**Example:**
```yaml
contact:
  type: email
```

### phone

Phone number field.

**Example:**
```yaml
telephone:
  type: phone
```

### address

Physical address field.

**Example:**
```yaml
location:
  type: address
```

### url

URL/website address.

**Example:**
```yaml
website:
  type: url
```

### array

List of items.
Arrays are typically populated using custom generators in the application code.


**Example:**
```yaml
items:
  type: array
```

### foreign_key

Reference to another table in the database.

**Example:**
```yaml
user_id:
  type: foreign_key
```




## Supported Field Properties


Fields can be defined with additional properties by using an object instead of a string:

```yaml
name:
  type: text
  description: Full name of the person
  constraints:
    max_length: 50
```
The following properties can be used when defining fields:


### description

Human-readable description of the field's purpose.

**Example:**
```yaml
email:
  type: email
  description: "User's email address for notifications"
```

### length

Fixed length for the field.

**Example:**
```yaml
zip_code:
  type: string
  length: 10
```

### min_length

Minimum length for string fields.

**Example:**
```yaml
password:
  type: string
  min_length: 8
```

### max_length

Maximum length for string fields.

**Example:**
```yaml
bio:
  type: text
  max_length: 250
```

### min

Minimum value for numeric fields.

**Example:**
```yaml
min: 0
```

### max

Maximum value for numeric fields.

**Example:**
```yaml
max: 1000
```

### decimals

Number of decimal places for float fields.

**Example:**
```yaml
price:
  type: float
  decimals: 2
```

### format

Format string specification (for dates, etc.).

**Example:**
```yaml
creation_date:
  type: date
  format: "YYYY-MM-DD"
```

### enum

List of possible values for the field.

**Example:**
```yaml
status:
  type: string
  enum: ["active", "inactive", "pending"]
```

### pattern

Regular expression pattern for validation.

**Example:**
```yaml
pattern: "^[A-Z][a-z]+$"
```

### references

Foreign key reference to another table and field.

**Example:**
```yaml
user_id:
  type: foreign_key
  references: {"schema": "User", "field": "id"}
```

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

The following constraints are supported:

### not_null

Field cannot be null or empty.

**Example:**
```yaml
username:
  type: string
  constraints:
    not_null: true
```

### unique

Values must be unique within the dataset.

**Example:**
```yaml
email:
  type: email
  constraints:
    unique: true
```

### primary_key

Field is a primary key for the table.

**Example:**
```yaml
id:
  type: integer
  constraints:
    primary_key: true
```

### format

Format specification for specialized field types.

**Example:**
```yaml
contact:
  type: string
  constraints:
    format: "email"
```

### pattern

Regular expression pattern for validation.

**Example:**
```yaml
pattern: "^[A-Z][a-z]+$"
```

### min

Minimum value for numeric fields.

**Example:**
```yaml
min: 0
```

### max

Maximum value for numeric fields.

**Example:**
```yaml
max: 1000
```

### min_length

Minimum string length for text fields.

**Example:**
```yaml
password:
  type: string
  constraints:
    min_length: 8
```

### max_length

Maximum string length for text fields.

**Example:**
```yaml
description:
  type: text
  constraints:
    max_length: 500
```

### length

Exact string length for text fields.

**Example:**
```yaml
country_code:
  type: string
  constraints:
    length: 2
```
