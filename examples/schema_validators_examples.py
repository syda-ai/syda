"""
Schema Validator Examples - Quick Usage Guide

This script demonstrates how to use the schema validators in SYDA.
Run this to see validation in action with various examples.
"""

from syda.validators import (
    SchemaValidator,
    ForeignKeyValidator,
    TemplateValidator,
    ConstraintValidator,
    CircularDependencyValidator,
    ValidationResult
)

# ============================================================================
# EXAMPLE 1: Basic Validation (Valid Schema)
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 1: Valid Schema Validation")
print("="*80)

valid_schemas = {
    'customers': {
        'id': 'integer',
        'name': 'text',
        'email': 'email'
    },
    'orders': {
        '__foreign_keys__': {
            'customer_id': ['customers', 'id']
        },
        'id': 'integer',
        'customer_id': 'foreign_key',
        'total': {
            'type': 'number',
            'constraints': {'min': 0, 'max': 999999}
        }
    }
}

validator = SchemaValidator()
result = validator.validate_schemas(valid_schemas)
print(result.summary())

# ============================================================================
# EXAMPLE 2: Foreign Key Error (Schema Not Found)
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Foreign Key Error - Schema Not Found")
print("="*80)

fk_error_schema = {
    'orders': {
        '__foreign_keys__': {
            'customer_id': ['customer', 'id']  # 'customer' doesn't exist!
        },
        'id': 'integer',
        'customer_id': 'foreign_key',
        'total': 'number'
    }
}

result = validator.validate_schemas(fk_error_schema)
print(result.summary())

# ============================================================================
# EXAMPLE 3: Constraint Error (Invalid Range)
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: Constraint Error - Invalid Range (min > max)")
print("="*80)

constraint_error_schema = {
    'products': {
        'id': 'integer',
        'price': {
            'type': 'number',
            'constraints': {
                'min': 1000,  # WRONG: min > max
                'max': 100
            }
        }
    }
}

result = validator.validate_schemas(constraint_error_schema)
print(result.summary())

# ============================================================================
# EXAMPLE 4: String Length Constraint Error
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Constraint Error - String Length (min > max)")
print("="*80)

string_constraint_error = {
    'users': {
        'id': 'integer',
        'username': {
            'type': 'text',
            'constraints': {
                'min_length': 100,  # WRONG: min_length > max_length
                'max_length': 10
            }
        }
    }
}

result = validator.validate_schemas(string_constraint_error)
print(result.summary())

# ============================================================================
# EXAMPLE 5: Invalid Regex Pattern
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 5: Constraint Error - Invalid Regex")
print("="*80)

regex_error_schema = {
    'products': {
        'sku': {
            'type': 'string',
            'constraints': {
                'pattern': '^[A-Z-[0-9]{5}$'  # WRONG: Unbalanced brackets
            }
        }
    }
}

result = validator.validate_schemas(regex_error_schema)
print(result.summary())

# ============================================================================
# EXAMPLE 6: Manual Foreign Key Validation
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 6: Manual Foreign Key Validation")
print("="*80)

schemas = {
    'customers': {'id': 'integer', 'name': 'text'},
    'orders': {
        '__foreign_keys__': {'customer_id': ['customers', 'id']},
        'id': 'integer',
        'customer_id': 'foreign_key'
    }
}

fk_validator = ForeignKeyValidator()
errors, warnings = fk_validator.validate_foreign_keys(
    'orders', schemas['orders'], schemas
)

print(f"Foreign Key Errors: {len(errors)}")
print(f"Foreign Key Warnings: {len(warnings)}")

if errors:
    for error in errors:
        print(f"  ❌ {error}")
else:
    print("  ✅ All foreign keys valid!")

if warnings:
    for warning in warnings:
        print(f"  ⚠️  {warning}")

# ============================================================================
# EXAMPLE 7: Strict Mode Validation
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 7: Strict Mode (Warnings become Errors)")
print("="*80)

schemas_with_warnings = {
    'customers': {'id': 'integer'},
    'orders': {
        '__foreign_keys__': {
            'cust_id': ['customers', 'id']  # Non-standard naming (warning)
        },
        'id': 'integer',
        'cust_id': 'foreign_key'
    }
}

print("Normal mode (warnings ignored):")
result = validator.validate_schemas(schemas_with_warnings, strict=False)
print(f"  Valid: {result.is_valid}, Errors: {result.error_count}, Warnings: {result.warning_count}")

print("\nStrict mode (warnings become errors):")
result = validator.validate_schemas(schemas_with_warnings, strict=True)
print(f"  Valid: {result.is_valid}, Errors: {result.error_count}")
print(result.summary())

# ============================================================================
# EXAMPLE 8: Detailed Error Analysis
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 8: Accessing Detailed Validation Results")
print("="*80)

complex_schema = {
    'categories': {
        'id': 'integer',
        'name': 'text'
    },
    'products': {
        '__foreign_keys__': {
            'category_id': ['categories', 'id'],
            'brand_id': ['brands', 'id']  # 'brands' doesn't exist!
        },
        'id': 'integer',
        'category_id': 'foreign_key',
        'brand_id': 'foreign_key',
        'price': {
            'type': 'number',
            'constraints': {'min': 5000, 'max': 100}  # WRONG: min > max
        }
    }
}

result = validator.validate_schemas(complex_schema)

print(f"Is Valid: {result.is_valid}")
print(f"Error Count: {result.error_count}")
print(f"Warning Count: {result.warning_count}")
print(f"\nDetailed Errors:")

for schema_name, errors in result.errors.items():
    if errors:
        print(f"\n  {schema_name}:")
        for error in errors:
            print(f"    ❌ {error}")

for schema_name, warnings in result.warnings.items():
    if warnings:
        print(f"\n  {schema_name}:")
        for warning in warnings:
            print(f"    ⚠️  {warning}")

if result.suggestions:
    print(f"\nSuggestions:")
    for suggestion in result.suggestions:
        print(f"  ✓ {suggestion}")

# ============================================================================
# EXAMPLE 9: Multi-Schema with Foreign Keys
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 9: E-Commerce Schema (Multi-level FKs)")
print("="*80)

ecommerce_schema = {
    'categories': {
        'id': 'integer',
        'name': 'text'
    },
    'products': {
        '__foreign_keys__': {
            'category_id': ['categories', 'id']
        },
        'id': 'integer',
        'category_id': 'foreign_key',
        'name': 'text',
        'price': {
            'type': 'number',
            'constraints': {'min': 0.01, 'max': 99999.99}
        }
    },
    'customers': {
        'id': 'integer',
        'email': 'email',
        'name': 'text'
    },
    'orders': {
        '__foreign_keys__': {
            'customer_id': ['customers', 'id']
        },
        'id': 'integer',
        'customer_id': 'foreign_key',
        'total': {
            'type': 'number',
            'constraints': {'min': 0.01, 'max': 999999.99}
        }
    },
    'order_items': {
        '__foreign_keys__': {
            'order_id': ['orders', 'id'],
            'product_id': ['products', 'id']
        },
        'id': 'integer',
        'order_id': 'foreign_key',
        'product_id': 'foreign_key',
        'quantity': {
            'type': 'integer',
            'constraints': {'min': 1, 'max': 1000}
        }
    }
}

result = validator.validate_schemas(ecommerce_schema)
print(result.summary())

# ============================================================================
# EXAMPLE 10: Accessing Individual Validators
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 10: Using Individual Validators")
print("="*80)

test_schema = {
    'price': {
        'type': 'number',
        'constraints': {'min': 10, 'max': 100}
    },
    'name': {
        'type': 'text',
        'constraints': {'min_length': 1, 'max_length': 50}
    }
}

constraint_validator = ConstraintValidator()
errors, warnings = constraint_validator.validate_constraints('products', test_schema)

print(f"Constraint Validation Results:")
print(f"  Errors: {len(errors)}")
print(f"  Warnings: {len(warnings)}")

if errors:
    for error in errors:
        print(f"    ❌ {error}")
else:
    print("  ✅ All constraints are valid!")

print("\n" + "="*80)
print("Examples completed! Check the output above for validation patterns.")
print("="*80)
