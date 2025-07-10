#!/usr/bin/env python3
"""
Schema Validation Testing Script

This script demonstrates schema validation with both valid and invalid schemas.
It showcases how validation errors are handled and reported to users.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syda.generate import SyntheticDataGenerator
import logging

# Set up logging to see validation messages
logging.basicConfig(level=logging.INFO)

def test_schema_validation():
    """Test loading and validating various schema files"""
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticDataGenerator()
    schemas_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
    
    # Test valid schemas
    valid_schemas = [
        "valid_customer.yml",
        "valid_product.yml"
    ]
    
    print("\n===== Testing Valid Schemas =====")
    for schema_file in valid_schemas:
        schema_path = os.path.join(schemas_dir, schema_file)
        try:
            print(f"\nValidating schema: {schema_file}")
            # Use generate_for_schemas to validate, but set default_sample_size to 1 to keep processing minimal
            schema_name = os.path.splitext(os.path.basename(schema_file))[0]
            results = generator.generate_for_schemas(
                schemas={schema_name: schema_path},
                output_dir=output_dir,
                default_sample_size=1  # Keep minimal to avoid long processing
            )
            print(f"✅ Schema validation successful: {schema_file}")
            table_name = os.path.splitext(os.path.basename(schema_file))[0].replace("valid_", "")
            print(f"   Schema name: {table_name}")
        except Exception as e:
            print(f"❌ Unexpected validation error: {e}")
    
    # Test invalid schemas
    invalid_schemas = [
        "invalid_missing_name.yml",
        "invalid_template_type.yml",
        "invalid_field_type.yml",
        "invalid_foreign_key.yml"
    ]
    
    print("\n===== Testing Invalid Schemas =====")
    for schema_file in invalid_schemas:
        schema_path = os.path.join(schemas_dir, schema_file)
        try:
            print(f"\nValidating schema: {schema_file}")
            # Try to generate with an invalid schema
            schema_name = os.path.splitext(os.path.basename(schema_file))[0]
            results = generator.generate_for_schemas(
                schemas={schema_name: schema_path},
                output_dir=output_dir,
                default_sample_size=1
            )
            print(f"❓ Unexpected success: {schema_file} - validation should have failed!")
        except Exception as e:
            # This is expected for invalid schemas
            print(f"✓ Validation correctly failed: {schema_file}")
            print(f"   Error: {str(e)}")

if __name__ == "__main__":
    print("Schema Validation Test")
    print("=====================")
    test_schema_validation()
    print("\nTest complete!")
