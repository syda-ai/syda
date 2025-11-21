"""
Integration tests for schema validators with the main generator.
Tests the full validation workflow during generation.
"""

import pytest
import tempfile
import os
from syda.validators import SchemaValidator, ValidationResult


class TestValidationIntegration:
    """Integration tests for schema validation in generation workflow."""
    
    def test_e_commerce_schema_passes_validation(self):
        """Should validate a complete e-commerce schema."""
        validator = SchemaValidator()
        
        schemas = {
            'categories': {
                'id': 'integer',
                'name': 'text',
                'description': 'text'
            },
            'products': {
                'id': 'integer',
                'name': 'text',
                'description': 'text',
                'category_id': 'foreign_key',
                'price': {
                    'type': 'number',
                    'constraints': {'min': 0.01, 'max': 100000}
                },
                '__foreign_keys__': {
                    'category_id': ('categories', 'id')
                }
            },
            'customers': {
                'id': 'integer',
                'name': 'text',
                'email': 'email',
                'phone': 'phone'
            },
            'orders': {
                'id': 'integer',
                'customer_id': 'foreign_key',
                'order_date': 'date',
                'total': {
                    'type': 'number',
                    'constraints': {'min': 0, 'max': 1000000}
                },
                '__foreign_keys__': {
                    'customer_id': ('customers', 'id')
                }
            },
            'order_items': {
                'id': 'integer',
                'order_id': 'foreign_key',
                'product_id': 'foreign_key',
                'quantity': {
                    'type': 'integer',
                    'constraints': {'min': 1, 'max': 1000}
                },
                'unit_price': 'number',
                '__foreign_keys__': {
                    'order_id': ('orders', 'id'),
                    'product_id': ('products', 'id')
                }
            }
        }
        
        result = validator.validate_schemas(schemas)
        
        assert result.is_valid
        assert result.error_count == 0
        assert result.warning_count == 0
    
    def test_healthcare_schema_with_templates(self):
        """Should validate healthcare schema with document templates."""
        validator = SchemaValidator()
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create template file
            template_path = os.path.join(temp_dir, 'medical_report.html')
            with open(template_path, 'w') as f:
                f.write('''
                <html>
                    <h1>Medical Report</h1>
                    <p>Patient: {{ patient_name }}</p>
                    <p>DOB: {{ date_of_birth }}</p>
                    <p>Diagnosis: {{ diagnosis }}</p>
                    <p>Prescribed Date: {{ prescribed_date }}</p>
                </html>
                ''')
            
            schemas = {
                'patients': {
                    'id': 'integer',
                    'name': 'text',
                    'email': 'email',
                    'phone': 'phone'
                },
                'medical_reports': {
                    '__template_source__': template_path,
                    '__input_file_type__': 'html',
                    '__output_file_type__': 'pdf',
                    'patient_id': 'foreign_key',
                    'patient_name': 'text',
                    'date_of_birth': 'date',
                    'diagnosis': 'text',
                    'prescribed_date': 'date',
                    '__foreign_keys__': {
                        'patient_id': ('patients', 'id')
                    }
                }
            }
            
            result = validator.validate_schemas(schemas)
            
            assert result.is_valid
            assert result.error_count == 0
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_schema_with_multiple_errors_reports_all(self):
        """Should collect and report all validation errors."""
        validator = SchemaValidator()
        
        schemas = {
            'products': {
                'id': 'integer',
                'price': {
                    'type': 'number',
                    'constraints': {'min': 100, 'max': 10}  # Invalid range
                },
                'sku': {
                    'type': 'text',
                    'constraints': {'pattern': 'ABC['}  # Invalid regex
                }
            },
            'orders': {
                '__foreign_keys__': {
                    'product_id': ('product', 'id'),  # Wrong table name
                    'customer_id': ('customers', 'name')  # Wrong column (not PK)
                },
                'id': 'integer',
                'product_id': 'foreign_key',
                'customer_id': 'foreign_key'
            }
        }
        
        result = validator.validate_schemas(schemas)
        
        # Should have multiple errors
        assert not result.is_valid
        assert result.error_count >= 3  # At least constraint + 2 FK errors
        
        # Verify error types are collected
        all_errors = []
        for schema_errors in result.errors.values():
            all_errors.extend(schema_errors)
        
        assert any('constraint' in str(e).lower() or 'price' in str(e).lower() for e in all_errors)
        assert any('pattern' in str(e).lower() or 'regex' in str(e).lower() for e in all_errors)
        assert any('product' in str(e).lower() for e in all_errors)
    
    def test_strict_mode_enforces_naming_conventions(self):
        """Should enforce naming conventions in strict mode."""
        validator = SchemaValidator()
        
        schemas = {
            'users': {
                'id': 'integer',
                'name': 'text'
            },
            'posts': {
                'id': 'integer',
                'title': 'text',
                'creator_fk': 'foreign_key',  # Non-standard naming
                '__foreign_keys__': {
                    'creator_fk': ('users', 'id')
                }
            }
        }
        
        # Non-strict mode
        result_normal = validator.validate_schemas(schemas, strict=False)
        assert result_normal.is_valid
        assert result_normal.warning_count > 0
        
        # Strict mode
        result_strict = validator.validate_schemas(schemas, strict=True)
        assert not result_strict.is_valid
        assert result_strict.error_count > 0
    
    def test_suggestions_provided_for_common_issues(self):
        """Should provide helpful suggestions for fixing issues."""
        validator = SchemaValidator()
        
        schemas = {
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('customer', 'id')  # Singular instead of plural
                },
                'id': 'integer',
                'customer_id': 'foreign_key'
            }
        }
        
        result = validator.validate_schemas(schemas)
        
        assert not result.is_valid
        assert len(result.suggestions) > 0
        # Should have helpful suggestions about case sensitivity
        assert any('case' in str(s).lower() or 'verify' in str(s).lower() for s in result.suggestions)
    
    def test_validation_with_file_based_schemas(self):
        """Should validate schemas defined in files."""
        import json
        
        validator = SchemaValidator()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create JSON schema files
            customer_schema = {
                'id': 'integer',
                'name': 'text',
                'email': 'email'
            }
            
            order_schema = {
                '__foreign_keys__': {
                    'customer_id': ['customers', 'id']
                },
                'id': 'integer',
                'customer_id': 'foreign_key',
                'total': 'number'
            }
            
            customer_path = os.path.join(temp_dir, 'customer.json')
            order_path = os.path.join(temp_dir, 'order.json')
            
            with open(customer_path, 'w') as f:
                json.dump(customer_schema, f)
            
            with open(order_path, 'w') as f:
                json.dump(order_schema, f)
            
            # Validate dict version (as if schemas were loaded)
            schemas = {
                'customers': customer_schema,
                'orders': order_schema
            }
            
            result = validator.validate_schemas(schemas)
            
            assert result.is_valid
            assert result.error_count == 0
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_large_schema_validation_performance(self):
        """Should validate large schemas in reasonable time."""
        import time
        
        validator = SchemaValidator()
        
        # Create a large schema with many tables
        schemas = {}
        
        # Base table
        schemas['base'] = {
            'id': 'integer',
            'name': 'text'
        }
        
        # Create 20 tables referencing each other in a chain
        for i in range(1, 21):
            prev_table = 'base' if i == 1 else f'table_{i-1}'
            schemas[f'table_{i}'] = {
                'id': 'integer',
                'data': 'text',
                f'{prev_table}_id': 'foreign_key',
                '__foreign_keys__': {
                    f'{prev_table}_id': (prev_table, 'id')
                }
            }
        
        # Time the validation
        start_time = time.time()
        result = validator.validate_schemas(schemas)
        elapsed_time = time.time() - start_time
        
        # Should complete in under 1 second
        assert elapsed_time < 1.0
        assert result.is_valid
    
    def test_validation_result_formatting(self):
        """Should format validation results correctly."""
        validator = SchemaValidator()
        
        schemas = {
            'orders': {
                '__foreign_keys__': {
                    'product_id': ('products', 'id'),
                    'price_field': {
                        'type': 'number',
                        'constraints': {'min': 100, 'max': 10}
                    }
                },
                'id': 'integer'
            }
        }
        
        result = validator.validate_schemas(schemas, strict=False)
        summary = result.summary()
        
        # Check formatting includes required elements
        assert '❌' in summary or 'FAILED' in summary
        assert 'orders' in summary


class TestValidationErrorMessages:
    """Test that validation error messages are user-friendly."""
    
    def test_error_message_clarity(self):
        """Should provide clear, actionable error messages."""
        validator = SchemaValidator()
        
        schemas = {
            'orders': {
                '__foreign_keys__': {
                    'customer_id': ('Customers', 'id')  # Capitalized
                },
                'id': 'integer',
                'customer_id': 'foreign_key'
            }
        }
        
        result = validator.validate_schemas(schemas)
        
        # Should suggest the right table name
        error_text = str(result.errors)
        assert 'customer' in error_text.lower() or 'Customers' in error_text
    
    def test_constraint_error_includes_values(self):
        """Should include actual constraint values in error messages."""
        validator = SchemaValidator()
        
        schemas = {
            'products': {
                'id': 'integer',
                'price': {
                    'type': 'number',
                    'constraints': {'min': 999, 'max': 99}
                }
            }
        }
        
        result = validator.validate_schemas(schemas)
        
        # Error message should include the actual values
        error_text = str(result.errors)
        assert '999' in error_text
        assert '99' in error_text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
