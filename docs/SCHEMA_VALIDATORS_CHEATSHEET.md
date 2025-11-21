# Schema Validators - Cheat Sheet

Quick reference guide for using SYDA schema validators.

## Quick Import

```python
from syda import SchemaValidator
from syda.validators import (
    ForeignKeyValidator,
    TemplateValidator,
    ConstraintValidator,
    CircularDependencyValidator
)
```

## Quick Validation

```python
# Validate schemas
validator = SchemaValidator()
result = validator.validate_schemas(schemas)

# Check result
if result.is_valid:
    print("✅ Valid!")
else:
    print("❌ Invalid!")
    print(result.summary())
```

## Validation Checklist

### Foreign Keys
- [ ] Target schema exists
- [ ] Target column exists
- [ ] FK field defined in source schema
- [ ] Naming follows convention: `{entity}_id`
- [ ] No circular dependencies

**Format:**
```python
'__foreign_keys__': {
    'customer_id': ['customers', 'id']
}
```

### Templates
- [ ] Template file exists
- [ ] All `{{ placeholders }}` defined
- [ ] All fields used in template
- [ ] Jinja2 syntax valid
- [ ] Metadata fields present

**Required Metadata:**
- `__template_source__`: Path to template file
- `__input_file_type__`: html, txt, rtf
- `__output_file_type__`: pdf, html, txt, rtf

### Constraints
- [ ] Field types recognized
- [ ] `min <= max` for numeric ranges
- [ ] `min_length <= max_length` for strings
- [ ] Regex patterns valid

**Numeric Example:**
```python
'price': {
    'type': 'number',
    'constraints': {'min': 0, 'max': 10000}
}
```

**String Example:**
```python
'name': {
    'type': 'text',
    'constraints': {'min_length': 1, 'max_length': 100}
}
```

### Circular Dependencies
- [ ] No cycles in foreign key relationships
- [ ] Dependency depth < 10 (recommended)

## Common Errors

### FK Schema Not Found
```
❌ FK: Field references non-existent schema 'customer'

✅ FIX: Check pluralization
'__foreign_keys__': {'customer_id': ['customers', 'id']}
```

### FK Column Not Found
```
❌ FK: Field references non-existent column 'customers.client_id'

✅ FIX: Verify column exists
'__foreign_keys__': {'customer_id': ['customers', 'id']}
```

### Constraint Range Invalid
```
❌ Constraint: Field 'price' has min (1000) > max (100)

✅ FIX: Ensure min ≤ max
'constraints': {'min': 10, 'max': 1000}
```

### Template Placeholder Missing
```
❌ Template: Placeholder '{{ tax_amount }}' not defined

✅ FIX: Add field to schema
'tax_amount': 'number'
```

### Invalid Regex
```
❌ Pattern: unterminated character set

✅ FIX: Balance brackets
'pattern': '^[A-Z]{3}-[0-9]{5}$'
```

## Validation Modes

### Normal Mode
```python
# Warnings don't block generation
result = validator.validate_schemas(schemas, strict=False)
```

### Strict Mode
```python
# Warnings become errors
result = validator.validate_schemas(schemas, strict=True)
```

## Accessing Results

```python
result = validator.validate_schemas(schemas)

# Status
result.is_valid           # Boolean
result.error_count        # Integer
result.warning_count      # Integer

# Details
result.errors             # Dict[schema, List[error]]
result.warnings           # Dict[schema, List[warning]]
result.suggestions        # List[suggestion]

# Output
result.summary()          # Formatted string
```

## Field Types

**Recognized Types:**
```
Numeric:   integer, number, float, decimal
Text:      text, string
Email:     email, phone, url
Date:      date, datetime, time
Boolean:   boolean, bool
Other:     json, dict, foreign_key, id, uuid
```

## Valid FK Formats

### Format 1: Tuple (recommended)
```python
'__foreign_keys__': {
    'customer_id': ('customers', 'id')
}
```

### Format 2: List
```python
'__foreign_keys__': {
    'customer_id': ['customers', 'id']
}
```

### Format 3: Dict
```python
'__foreign_keys__': {
    'customer_id': {
        'schema': 'customers',
        'column': 'id'
    }
}
```

## Manual Validator Usage

### Foreign Key Validator
```python
fk_val = ForeignKeyValidator()
errors, warnings = fk_val.validate_foreign_keys(
    'orders', schemas['orders'], schemas
)
```

### Template Validator
```python
template_val = TemplateValidator()
errors, warnings = template_val.validate_templates(
    'invoices', schemas['invoices']
)
```

### Constraint Validator
```python
constraint_val = ConstraintValidator()
errors, warnings = constraint_val.validate_constraints(
    'products', schemas['products']
)
```

### Circular Dependency Validator
```python
circular_val = CircularDependencyValidator()
errors, warnings = circular_val.validate_circular_dependencies(
    'users', schemas['users'], schemas, max_depth=10
)
```

## Best Practices

1. **Use explicit FKs**
   ```python
   '__foreign_keys__': {'customer_id': ['customers', 'id']}
   ```

2. **Follow naming conventions**
   ```python
   'customer_id'      # ✅ Good
   'cust_id'          # ⚠️  Less clear
   'customer_fk'      # ❌ Non-standard
   ```

3. **Validate early**
   ```python
   # Before generation
   result = validator.validate_schemas(schemas)
   if result.is_valid:
       generate_data()
   ```

4. **Use constraints**
   ```python
   'price': {
       'type': 'number',
       'constraints': {'min': 0.01, 'max': 9999.99}
   }
   ```

5. **Check templates separately**
   ```python
   template_val.validate_templates('invoices', schema)
   ```

## Integration with Generation

```python
from syda import SyntheticDataGenerator
from syda.validators import SchemaValidator

# Validate
validator = SchemaValidator()
result = validator.validate_schemas(schemas)

# Generate if valid
if result.is_valid:
    gen = SyntheticDataGenerator()
    results = gen.generate_for_schemas(schemas=schemas)
else:
    print(result.summary())
```

## Performance

| Operation | Time |
|-----------|------|
| FK validation (10 schemas) | ~5ms |
| Template validation (5) | ~10ms |
| Constraint validation (50 fields) | ~2ms |
| **Total** | **<20ms** |

**Note:** Validation is <1% of total generation time

## Troubleshooting

**Q: Why does validator reject my schema?**
A: Run validation manually to see detailed errors:
```python
result = validator.validate_schemas(schemas)
print(result.summary())
```

**Q: How do I validate just one schema?**
A: Pass single schema dict:
```python
single = {'products': {...}}
result = validator.validate_schemas(single)
```

**Q: Can I validate without generating?**
A: Yes, manual validation is separate:
```python
result = validator.validate_schemas(schemas)
# Check result without calling generate_for_schemas()
```

**Q: Validation passes but generation fails?**
A: Validation covers FK, templates, constraints. Generation issues may be LLM-specific.

## Examples

### ✅ Valid Schema
```python
schemas = {
    'users': {
        'id': 'integer',
        'name': 'text'
    },
    'posts': {
        '__foreign_keys__': {'user_id': ['users', 'id']},
        'id': 'integer',
        'user_id': 'foreign_key',
        'title': 'text'
    }
}

result = validator.validate_schemas(schemas)
# ✅ All schemas passed validation!
```

### ❌ Invalid Schema
```python
schemas = {
    'posts': {
        '__foreign_keys__': {'user_id': ['user', 'id']},  # ❌ 'user' doesn't exist
        'id': 'integer',
        'user_id': 'foreign_key'
    }
}

result = validator.validate_schemas(schemas)
# ❌ SCHEMA VALIDATION FAILED
```

## Links

- 📖 [Full Documentation](../deep_dive/schema_reference.md)
- 🔗 [Foreign Keys](../deep_dive/foreign_keys.md)
- 🎨 [Templates](../deep_dive/unstructured_documents.md)
- 📝 [Field Types](../schema_reference/field_types.md)
- 💻 [Examples](./schema_validators_usage.md)
