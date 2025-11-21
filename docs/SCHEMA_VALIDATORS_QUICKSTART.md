# Schema Validators - Getting Started

Welcome! This guide will help you quickly learn how to use SYDA's schema validators.

## What Are Schema Validators?

Schema validators check your data schema **before** generating data, catching errors like:
- ❌ Foreign keys referencing non-existent tables
- ❌ Invalid constraint ranges (min > max)
- ❌ Template placeholders not defined
- ❌ Circular dependencies

**Why?** Finding these errors before AI generation saves 2-3 minutes of wasted API calls!

## Quick Start (30 seconds)

### Import
```python
from syda.validators import SchemaValidator
```

### Use
```python
validator = SchemaValidator()
result = validator.validate_schemas(schemas)

if result.is_valid:
    print("✅ Ready to generate!")
else:
    print(result.summary())
```

### Example
```python
schemas = {
    'customers': {'id': 'integer', 'name': 'text'},
    'orders': {
        '__foreign_keys__': {'customer_id': ['customers', 'id']},
        'id': 'integer',
        'customer_id': 'foreign_key'
    }
}

result = validator.validate_schemas(schemas)
print(result.summary())
# ✅ All schemas passed validation!
```

## The Two Ways to Use Validators

### 1. Automatic (Happens Automatically)
```python
from syda import SyntheticDataGenerator

generator = SyntheticDataGenerator()

# Validation runs automatically
results = generator.generate_for_schemas(schemas=schemas)
```

**When:** Always - validation runs silently in the background
**Benefit:** Catch errors before expensive AI calls

### 2. Manual (Pre-check Your Schema)
```python
from syda.validators import SchemaValidator

validator = SchemaValidator()
result = validator.validate_schemas(schemas)

if result.is_valid:
    # Then generate
else:
    # Fix errors and retry
```

**When:** Before generation or during schema development
**Benefit:** Get detailed error messages and suggestions

## Common Errors (and How to Fix Them)

### Error 1: Foreign Key References Wrong Schema
```
FK: Field 'customer_id' references non-existent schema 'customer'
```

**Fix:** Check the schema name (usually missing plural)
```python
# ❌ WRONG
'__foreign_keys__': {'customer_id': ['customer', 'id']}

# ✅ RIGHT
'__foreign_keys__': {'customer_id': ['customers', 'id']}
```

### Error 2: Constraint Range is Backwards
```
Constraint: Field 'price' has min (1000) > max (100)
```

**Fix:** Ensure min ≤ max
```python
# ❌ WRONG
'constraints': {'min': 1000, 'max': 100}

# ✅ RIGHT
'constraints': {'min': 10, 'max': 1000}
```

### Error 3: Template Placeholder Not Defined
```
Template: Placeholder '{{ invoice_id }}' is not defined in schema
```

**Fix:** Add the field to the schema
```python
# ❌ WRONG
schemas = {
    'invoices': {
        'customer_name': 'text',
        '__template_source__': 'templates/invoice.html'
    }
}

# ✅ RIGHT
schemas = {
    'invoices': {
        'invoice_id': 'integer',  # Add this!
        'customer_name': 'text',
        '__template_source__': 'templates/invoice.html'
    }
}
```

## Understanding Results

```python
result = validator.validate_schemas(schemas)

# Check status
result.is_valid           # True or False
result.error_count        # Number of errors
result.warning_count      # Number of warnings

# Get formatted output
print(result.summary())

# Access details
result.errors             # Dict of errors by schema
result.warnings           # Dict of warnings by schema
result.suggestions        # List of fix suggestions
```

## Validation Modes

### Normal Mode (Default)
```python
result = validator.validate_schemas(schemas)
# Warnings don't block generation
```

### Strict Mode
```python
result = validator.validate_schemas(schemas, strict=True)
# Warnings become errors - stricter checking
```

## Running Examples

See validators in action:

```bash
python examples/schema_validators_examples.py
```

This shows 10 different validation scenarios you can learn from.

## Import Methods

### Option 1: From Main Syda Module
```python
from syda import SchemaValidator
```

### Option 2: From Validators Submodule
```python
from syda.validators import SchemaValidator
```

### Option 3: Import Multiple Validators
```python
from syda.validators import (
    SchemaValidator,
    ForeignKeyValidator,
    TemplateValidator,
    ConstraintValidator
)
```

## When to Use Each Validator

| Validator | Use When | Problem It Catches |
|-----------|----------|-------------------|
| **SchemaValidator** | Always | All validation checks |
| **ForeignKeyValidator** | Debugging FKs | FK reference issues |
| **TemplateValidator** | Using templates | Missing placeholders, bad syntax |
| **ConstraintValidator** | Complex constraints | Invalid ranges, regex errors |
| **CircularDependencyValidator** | Complex schemas | Circular FK relationships |

## Next Steps

### For Quick Reference
→ Check `docs/SCHEMA_VALIDATORS_CHEATSHEET.md`

### For Detailed Learning
→ Read `docs/examples/schema_validators_usage.md`

### For Docstring Help
→ Read docstrings in `syda/validators.py`

### For Examples
→ Run `python examples/schema_validators_examples.py`

## Integration with Generation

```python
from syda import SyntheticDataGenerator
from syda.validators import SchemaValidator

# Step 1: Validate
validator = SchemaValidator()
result = validator.validate_schemas(schemas)

# Step 2: Check result
if result.is_valid:
    # Step 3: Generate
    generator = SyntheticDataGenerator()
    results = generator.generate_for_schemas(schemas=schemas)
    print("✅ Data generated!")
else:
    # Show what to fix
    print("Fix these errors:")
    print(result.summary())
```

## Performance Notes

- Validation: <20ms
- AI generation: 2000-5000ms
- **Overhead: <1%** of total time

Validation adds almost no overhead!

## Troubleshooting

### Q: How do I see detailed errors?
**A:** The validation report shows exactly what's wrong:
```python
result = validator.validate_schemas(schemas)
print(result.summary())
```

### Q: How do I validate just one schema?
**A:** Pass a single-schema dict:
```python
single = {'products': {...}}
result = validator.validate_schemas(single)
```

### Q: Can I validate specific aspects only?
**A:** Use individual validators:
```python
fk_validator = ForeignKeyValidator()
errors, warnings = fk_validator.validate_foreign_keys(
    'orders', schemas['orders'], schemas
)
```

### Q: How do I make validation stricter?
**A:** Use strict mode:
```python
result = validator.validate_schemas(schemas, strict=True)
```

## Summary

- ✅ **Import:** `from syda.validators import SchemaValidator`
- ✅ **Use:** `result = validator.validate_schemas(schemas)`
- ✅ **Check:** `if result.is_valid: ...`
- ✅ **Debug:** `print(result.summary())`

You're ready to validate schemas! For detailed information, check the additional documentation:
- Quick Reference: `docs/SCHEMA_VALIDATORS_CHEATSHEET.md`
- Full Guide: `docs/examples/schema_validators_usage.md`
- Runnable Examples: `examples/schema_validators_examples.py`

---

**Have Questions?**
1. Check the appropriate documentation file
2. Read the docstrings in `syda/validators.py`
3. Run the example script to see it in action
4. Review similar examples in `docs/examples/schema_validators_usage.md`
