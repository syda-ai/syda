"""
Syda - Synthetic Data Generation Library

A Python library for AI-powered synthetic data generation with referential integrity.
Supports multiple AI providers (OpenAI, Anthropic) and various schema formats.

BASIC USAGE
===========

Quick start with synthetic data generation:

    from syda import SyntheticDataGenerator, ModelConfig
    
    generator = SyntheticDataGenerator(
        model_config=ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-haiku-20241022"
        )
    )
    
    schemas = {
        'customers': {'id': 'integer', 'name': 'text'},
        'orders': {
            '__foreign_keys__': {'customer_id': ['customers', 'id']},
            'id': 'integer',
            'customer_id': 'foreign_key'
        }
    }
    
    results = generator.generate_for_schemas(schemas=schemas)

SCHEMA VALIDATION
=================

Validate schemas before generation or manually:

    from syda.validators import SchemaValidator
    
    validator = SchemaValidator()
    result = validator.validate_schemas(schemas)
    
    if result.is_valid:
        print("✅ Schemas are valid!")
    else:
        print(result.summary())

For more details on validation, see:
    - syda.validators.SchemaValidator
    - syda.validators.ForeignKeyValidator
    - syda.validators.TemplateValidator
    - syda.validators.ConstraintValidator
    - syda.validators.CircularDependencyValidator
"""

from .generate import SyntheticDataGenerator
from .schemas import ModelConfig
from .validators import (
    SchemaValidator,
    ValidationResult,
    ForeignKeyValidator,
    TemplateValidator,
    ConstraintValidator,
    CircularDependencyValidator
)

__all__ = [
    'SyntheticDataGenerator',
    'ModelConfig',
    'SchemaValidator',
    'ValidationResult',
    'ForeignKeyValidator',
    'TemplateValidator',
    'ConstraintValidator',
    'CircularDependencyValidator'
]

__version__ = '0.0.4'
__author__ = 'Rama Krishna Kumar Lingamgunta'
__email__ = 'ramkumar2606@gmail.com'
__license__ = 'MIT'
__description__ = 'Seamlessly generates realistic synthetic test data—including structured, unstructured, PDF, and HTML—using AI and large language models. It preserves referential integrity, maintains privacy compliance, and accelerates development workflows. SYDA enables both highly regulated industries such as healthcare and banking, as well as non-regulated environments like software testing, research, and analytics, to safely simulate diverse data scenarios without exposing sensitive information.'
