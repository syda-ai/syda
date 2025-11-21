"""
Unit tests for the schema validators module.
"""

import pytest
import os
import tempfile
from pathlib import Path

from syda.validators import (
    SchemaValidator,
    ForeignKeyValidator,
    TemplateValidator,
    ConstraintValidator,
    CircularDependencyValidator,
    ValidationResult
)


class TestForeignKeyValidator:
    """Test cases for ForeignKeyValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ForeignKeyValidator()
    
    def test_valid_foreign_key(self):
        """Should pass validation for valid FK."""
        schemas = {
            'customers': {
                'id': 'integer',
                'name': 'text'
            },
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('customers', 'id')
                },
                'id': 'integer',
                'customer_id': 'foreign_key',
                'total': 'number'
            }
        }
        
        errors, warnings = self.validator.validate_foreign_keys(
            'orders', schemas['orders'], schemas
        )
        
        assert len(errors) == 0, f"Expected no errors, got: {errors}"
    
    def test_missing_target_schema(self):
        """Should error when FK references non-existent schema."""
        schemas = {
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('customer', 'id')  # Wrong name
                },
                'id': 'integer',
                'customer_id': 'foreign_key'
            }
        }
        
        errors, warnings = self.validator.validate_foreign_keys(
            'orders', schemas['orders'], schemas
        )
        
        assert len(errors) > 0
        assert any('customer' in str(e).lower() for e in errors)
    
    def test_missing_target_column(self):
        """Should error when FK references non-existent column."""
        schemas = {
            'customers': {
                'id': 'integer',
                'name': 'text'
            },
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('customers', 'customer_id')  # Wrong column
                },
                'id': 'integer',
                'customer_id': 'foreign_key'
            }
        }
        
        errors, warnings = self.validator.validate_foreign_keys(
            'orders', schemas['orders'], schemas
        )
        
        assert len(errors) > 0
        assert any('column' in str(e).lower() for e in errors)
    
    def test_naming_convention_warning(self):
        """Should warn for non-standard FK naming."""
        schemas = {
            'customers': {
                'id': 'integer',
                'name': 'text'
            },
            'orders': {
                '__foreign_keys__': {
                    'cust_fk': ('customers', 'id')  # Non-standard name
                },
                'id': 'integer',
                'cust_fk': 'foreign_key'
            }
        }
        
        errors, warnings = self.validator.validate_foreign_keys(
            'orders', schemas['orders'], schemas
        )
        
        assert len(errors) == 0
        assert len(warnings) > 0
        assert any('naming convention' in str(w).lower() for w in warnings)
    
    def test_missing_fk_field_in_schema(self):
        """Should error when FK field not defined in schema."""
        schemas = {
            'customers': {
                'id': 'integer'
            },
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('customers', 'id')
                },
                'id': 'integer'
                # customer_id not defined here
            }
        }
        
        errors, warnings = self.validator.validate_foreign_keys(
            'orders', schemas['orders'], schemas
        )
        
        assert len(errors) > 0
        assert any('not defined' in str(e).lower() for e in errors)


class TestTemplateValidator:
    """Test cases for TemplateValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TemplateValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_template_file(self):
        """Should error when template file doesn't exist."""
        schema = {
            '__template_source__': '/nonexistent/path/template.html',
            '__input_file_type__': 'html',
            '__output_file_type__': 'pdf',
            'name': 'text'
        }
        
        errors, warnings = self.validator.validate_templates('invoice', schema)
        
        assert len(errors) > 0
        assert any('not found' in str(e).lower() for e in errors)
    
    def test_missing_placeholder_in_schema(self):
        """Should error when template has placeholder not in schema."""
        # Create temporary template file
        template_path = os.path.join(self.temp_dir, 'template.html')
        with open(template_path, 'w') as f:
            f.write('<p>Hello {{ customer_name }}</p>')
        
        schema = {
            '__template_source__': template_path,
            '__input_file_type__': 'html',
            '__output_file_type__': 'html',
            'invoice_id': 'text'
            # customer_name not in schema
        }
        
        errors, warnings = self.validator.validate_templates('invoice', schema)
        
        assert len(errors) > 0
        assert any('customer_name' in str(e).lower() for e in errors)
    
    def test_valid_template(self):
        """Should pass validation for valid template."""
        # Create temporary template file
        template_path = os.path.join(self.temp_dir, 'template.html')
        with open(template_path, 'w') as f:
            f.write('<p>Invoice: {{ invoice_id }}, Customer: {{ customer_name }}</p>')
        
        schema = {
            '__template_source__': template_path,
            '__input_file_type__': 'html',
            '__output_file_type__': 'html',
            'invoice_id': 'text',
            'customer_name': 'text'
        }
        
        errors, warnings = self.validator.validate_templates('invoice', schema)
        
        assert len(errors) == 0, f"Expected no errors, got: {errors}"
    
    def test_missing_template_metadata(self):
        """Should error when required template metadata is missing."""
        # Create temporary template file
        template_path = os.path.join(self.temp_dir, 'template.html')
        with open(template_path, 'w') as f:
            f.write('<p>{{ name }}</p>')
        
        schema = {
            '__template_source__': template_path,
            # Missing __input_file_type__ and __output_file_type__
            'name': 'text'
        }
        
        errors, warnings = self.validator.validate_templates('invoice', schema)
        
        assert len(errors) >= 2  # Both metadata fields missing
        assert any('input_file_type' in str(e).lower() for e in errors)
        assert any('output_file_type' in str(e).lower() for e in errors)
    
    def test_non_template_schema(self):
        """Should skip validation for non-template schemas."""
        schema = {
            'id': 'integer',
            'name': 'text'
        }
        
        errors, warnings = self.validator.validate_templates('customer', schema)
        
        assert len(errors) == 0
        assert len(warnings) == 0


class TestConstraintValidator:
    """Test cases for ConstraintValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConstraintValidator()
    
    def test_invalid_numeric_range(self):
        """Should error when min > max."""
        schema = {
            'price': {
                'type': 'number',
                'constraints': {
                    'min': 1000,
                    'max': 100
                }
            }
        }
        
        errors, warnings = self.validator.validate_constraints('product', schema)
        
        assert len(errors) > 0
        assert any('min' in str(e).lower() and 'max' in str(e).lower() for e in errors)
    
    def test_invalid_regex_pattern(self):
        """Should error for invalid regex patterns."""
        schema = {
            'sku': {
                'type': 'text',
                'constraints': {
                    'pattern': 'invalid['  # Invalid regex
                }
            }
        }
        
        errors, warnings = self.validator.validate_constraints('product', schema)
        
        assert len(errors) > 0
        assert any('regex' in str(e).lower() or 'pattern' in str(e).lower() for e in errors)
    
    def test_invalid_string_length_range(self):
        """Should error when min_length > max_length."""
        schema = {
            'name': {
                'type': 'text',
                'constraints': {
                    'min_length': 100,
                    'max_length': 10
                }
            }
        }
        
        errors, warnings = self.validator.validate_constraints('product', schema)
        
        assert len(errors) > 0
        assert any('length' in str(e).lower() for e in errors)
    
    def test_valid_constraints(self):
        """Should pass validation for valid constraints."""
        schema = {
            'price': {
                'type': 'number',
                'constraints': {
                    'min': 0,
                    'max': 10000
                }
            },
            'name': {
                'type': 'text',
                'constraints': {
                    'pattern': '^[A-Z][a-z]+$',
                    'min_length': 1,
                    'max_length': 100
                }
            }
        }
        
        errors, warnings = self.validator.validate_constraints('product', schema)
        
        assert len(errors) == 0, f"Expected no errors, got: {errors}"
    
    def test_unknown_field_type_warning(self):
        """Should warn for unknown field types."""
        schema = {
            'custom_field': 'unknown_type'
        }
        
        errors, warnings = self.validator.validate_constraints('product', schema)
        
        assert len(warnings) > 0
        assert any('unknown' in str(w).lower() or 'type' in str(w).lower() for w in warnings)


class TestSchemaValidator:
    """Test cases for main SchemaValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SchemaValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_valid_schemas(self):
        """Should validate correct schema definitions."""
        schemas = {
            'customers': {
                'id': 'integer',
                'name': 'text',
                'email': 'email'
            },
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('customers', 'id')
                },
                'id': 'integer',
                'customer_id': 'foreign_key',
                'total': 'number'
            }
        }
        
        result = self.validator.validate_schemas(schemas)
        
        assert result.is_valid
        assert result.error_count == 0
    
    def test_empty_schemas(self):
        """Should error for empty schemas."""
        schemas = {}
        
        result = self.validator.validate_schemas(schemas)
        
        assert not result.is_valid
        assert result.error_count > 0
    
    def test_schema_with_no_fields(self):
        """Should error for schema with only metadata."""
        schemas = {
            'orders': {
                '__foreign_keys__': {},
                '__table_name__': 'orders'
                # No actual data fields
            }
        }
        
        result = self.validator.validate_schemas(schemas)
        
        assert not result.is_valid
        assert any('at least one' in str(e).lower() for errors in result.errors.values() for e in errors)
    
    def test_multiple_validation_errors(self):
        """Should report all validation errors."""
        schemas = {
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('customer', 'id'),  # Wrong table
                    'product_id': ('products', 'product_id')  # Wrong column
                },
                'id': 'integer'
            }
        }
        
        result = self.validator.validate_schemas(schemas)
        
        assert not result.is_valid
        assert result.error_count >= 2
    
    def test_strict_mode_converts_warnings_to_errors(self):
        """Should convert warnings to errors in strict mode."""
        schemas = {
            'customers': {
                'id': 'integer',
                'name': 'text'
            },
            'orders': {
                '__foreign_keys__': {
                    'cust_fk': ('customers', 'id')  # Non-standard naming
                },
                'id': 'integer',
                'cust_fk': 'foreign_key'
            }
        }
        
        # Non-strict mode
        result_normal = self.validator.validate_schemas(schemas, strict=False)
        normal_has_warning = result_normal.warning_count > 0
        
        # Strict mode
        result_strict = self.validator.validate_schemas(schemas, strict=True)
        
        assert normal_has_warning, "Should have warning in non-strict mode"
        assert not result_strict.is_valid, "Should be invalid in strict mode"
        assert result_strict.error_count > 0, "Should have errors converted from warnings"
    
    def test_circular_dependencies_detection(self):
        """Should detect circular foreign key dependencies."""
        schemas = {
            'a': {
                '__foreign_keys__': {
                    'b_id': ('b', 'id')
                },
                'id': 'integer',
                'b_id': 'foreign_key'
            },
            'b': {
                '__foreign_keys__': {
                    'a_id': ('a', 'id')  # Circular: a -> b -> a
                },
                'id': 'integer',
                'a_id': 'foreign_key'
            }
        }
        
        result = self.validator.validate_schemas(schemas)
        
        # Should detect the circular dependency
        assert not result.is_valid
        assert result.error_count > 0
        assert any('circular' in str(e).lower() for errors in result.errors.values() for e in errors)


class TestValidationResult:
    """Test cases for ValidationResult data class."""
    
    def test_add_error(self):
        """Should add errors and mark as invalid."""
        result = ValidationResult()
        
        result.add_error('schema1', 'Error message')
        
        assert not result.is_valid
        assert result.error_count == 1
        assert 'schema1' in result.errors
        assert 'Error message' in result.errors['schema1']
    
    def test_add_warning(self):
        """Should add warnings."""
        result = ValidationResult()
        
        result.add_warning('schema1', 'Warning message')
        
        assert result.is_valid  # Warnings don't make it invalid
        assert result.warning_count == 1
        assert 'schema1' in result.warnings
    
    def test_add_suggestion(self):
        """Should add suggestions."""
        result = ValidationResult()
        
        result.add_suggestion('Fix this issue')
        result.add_suggestion('Fix this issue')  # Duplicate
        
        assert len(result.suggestions) == 1  # Should not add duplicates
    
    def test_summary_formatting(self):
        """Should format summary correctly."""
        result = ValidationResult()
        result.add_error('schema1', 'Error 1')
        result.add_warning('schema1', 'Warning 1')
        result.add_suggestion('Suggestion 1')
        
        summary = result.summary()
        
        assert 'FAILED' in summary
        assert 'Error 1' in summary
        assert 'Warning 1' in summary
        assert 'Suggestion 1' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
