#!/usr/bin/env python
"""
Example demonstrating dictionary schema validation in the SYDA project.

This script includes:
1. Valid dictionary schemas as a reference
2. Various invalid dictionary schemas demonstrating common validation errors
3. Testing validation for both valid and invalid schemas
4. Clear error reporting with schema details

The test validates schemas directly by using the SyntheticDataGenerator
interface, matching the actual usage in production code.
"""

import sys
import os
import random
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()

# Import the synthetic data generator
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

def main():
    """Test schema validation with various valid and invalid dictionary schemas."""
    
    print("Dictionary Schema Validation Test")
    print("================================\n")
    
    # Create a generator instance with appropriate settings
    model_config = ModelConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        temperature=0.5,
        max_tokens=2048,
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Define output directory (won't be used for invalid schemas)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================
    # Define VALID dictionary schemas for reference
    # =========================================================
    valid_schemas = {
        # Valid Customer schema
        'valid_customer': {
            '__name__': 'Customer',
            '__table_description__': 'Valid customer schema for testing',
            'id': {'type': 'integer', 'description': 'Unique identifier', 'primary_key': True},
            'name': {'type': 'string', 'description': 'Customer name'},
            'email': {'type': 'email', 'description': 'Customer email'}
        },
        
        # Valid Product schema
        'valid_product': {
            '__name__': 'Product',
            '__table_description__': 'Valid product schema for testing',
            'id': {'type': 'integer', 'description': 'Unique identifier', 'primary_key': True},
            'name': {'type': 'string', 'description': 'Product name'},
            'price': {'type': 'float', 'description': 'Product price'}
        }
    }
    
    # =========================================================
    # Define INVALID dictionary schemas with various errors
    # =========================================================
    invalid_schemas = {
        # Missing required __name__ field
        'invalid_missing_name': {
            # __name__ is missing
            '__table_description__': 'Schema missing required __name__ field',
            'id': {'type': 'integer', 'description': 'Unique identifier'},
            'title': {'type': 'string', 'description': 'Title field'}
        },
        
        # Invalid field type
        'invalid_field_type': {
            '__name__': 'InvalidFieldType',
            '__table_description__': 'Schema with invalid field type',
            'id': {'type': 'integer', 'description': 'Unique identifier'},
            'age': {'type': 'non_existent_type', 'description': 'Age with invalid type'}
        },
        
        # Invalid template schema
        'invalid_template': {
            '__name__': 'InvalidTemplate',
            '__template__': 'not-a-boolean',  # Should be boolean
            '__table_description__': 'Schema with invalid template field',
            'id': {'type': 'integer', 'description': 'Unique identifier'},
            'name': {'type': 'string', 'description': 'Name field'}
            # Missing required template_source for template schema
        },
        
        # Invalid foreign key reference
        'invalid_foreign_key': {
            '__name__': 'InvalidForeignKey',
            '__table_description__': 'Schema with invalid foreign key reference',
            '__foreign_keys__': {
                'missing_column': ['NonExistentTable', 'id'],  # References non-existent table
                'product_id': ['Product', 'non_existent_field']  # References non-existent field
            },
            'id': {'type': 'integer', 'description': 'Unique identifier'},
            'product_id': {'type': 'foreign_key', 'description': 'Foreign key with invalid reference'}
        }
    }
    
    # =========================================================
    # Test VALID schemas
    # =========================================================
    print("\n===== Testing Valid Dictionary Schemas =====\n")
    
    for schema_name, schema in valid_schemas.items():
        print(f"Validating schema: {schema_name}")
        try:
            # Use generate_for_schemas to validate the schema through the user-facing API
            schemas_dict = {schema_name: schema}
            results = generator.generate_for_schemas(
                schemas=schemas_dict,
                output_dir=output_dir,
                default_sample_size=1  # Minimal to avoid long processing
            )
            print(f"✅ Schema validation successful: {schema_name}")
            print(f"   Schema name: {schema.get('__name__', 'Unknown')}")
        except ValueError as e:
            print(f"❌ Unexpected validation error: {str(e)}")
    
    # =========================================================
    # Test INVALID schemas
    # =========================================================
    print("\n===== Testing Invalid Dictionary Schemas =====\n")
    
    for schema_name, schema in invalid_schemas.items():
        print(f"Validating schema: {schema_name}")
        try:
            # Use generate_for_schemas to validate the schema through the user-facing API
            schemas_dict = {schema_name: schema}
            results = generator.generate_for_schemas(
                schemas=schemas_dict,
                output_dir=output_dir,
                default_sample_size=1
            )
            print(f"❓ Unexpected success: {schema_name} - validation should have failed!")
        except ValueError as e:
            print(f"✓ Validation correctly failed: {schema_name}")
            print(f"   Error: {str(e)}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
