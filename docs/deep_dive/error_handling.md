# Error Handling and Troubleshooting

Effective error handling is crucial when generating synthetic data at scale. This guide covers common error scenarios in SYDA and how to troubleshoot and resolve them.

## Common Error Types

### Schema Validation Errors

Schema validation errors occur when your schema definition doesn't meet SYDA's requirements:

```python
# Missing required field type
schemas = {
    'Customer': {
        'id': {'primary_key': True},  # Missing 'type' property
        'name': {'type': 'string'}
    }
}
```

**Resolution**: Ensure all fields have at least a `type` property and that required special attributes are included.

### Foreign Key Errors

Foreign key errors happen when references are invalid or circular:

```
ValueError: Circular dependency detected in schema: 'OrderItem' → 'Order' → 'Customer' → 'PreferredStore' → 'OrderItem'
```

**Resolution**: Check your schema for circular references and ensure that all foreign key references point to valid tables and columns.

### Template Processing Errors

Template processing errors occur when Jinja2 template rendering fails:

```
jinja2.exceptions.UndefinedError: 'discount_amount' is undefined
```

**Resolution**:
1. Check that all variables used in templates are properly defined
2. Use the `default` filter for optional variables: `{{ discount_amount | default(0) }}`
3. When embedding templates in Markdown, wrap them in `{% raw %}` and `{% endraw %}` tags

### Model API Errors

Model API errors happen when communication with the LLM provider fails:

```
anthropic.RateLimitError: Rate limit exceeded, please retry after 5s
```

**Resolution**:
1. Implement proper error handling with exponential backoff
2. Consider using a different model or provider
3. Monitor API usage and adjust batch sizes accordingly

## Implementing Error Handling

Here's an example of robust error handling when generating data:

```python
from syda import SyntheticDataGenerator, ModelConfig
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('syda')

def generate_with_retry(generator, schemas, sample_sizes, max_retries=3):
    """Generate data with retry logic for API errors"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            results = generator.generate_for_schemas(
                schemas=schemas,
                sample_sizes=sample_sizes,
                output_dir="output/data"
            )
            return results
        except Exception as e:
            retry_count += 1
            wait_time = 2 ** retry_count  # Exponential backoff
            
            if "Rate limit" in str(e) and retry_count < max_retries:
                logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif retry_count < max_retries:
                logger.error(f"Error during generation: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
                
# Usage
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)

try:
    results = generate_with_retry(generator, schemas, {"Customer": 10, "Order": 25})
    logger.info("Generation completed successfully")
except Exception as e:
    logger.critical(f"Generation failed completely: {str(e)}")
```

## Schema Validation

SYDA performs validation before generating data to catch common issues:

```python
from syda import SyntheticDataGenerator, ModelConfig, validate_schemas

# Validate schemas before generation
schemas = {
    'Customer': {
        'id': {'type': 'integer', 'primary_key': True},
        'name': {'type': 'string'}
    },
    'Order': {
        'id': {'type': 'integer', 'primary_key': True},
        'customer_id': {
            'type': 'integer',
            'references': {'table': 'Customer', 'column': 'id'}
        }
    }
}

try:
    # Perform validation only
    validate_schemas(schemas)
    print("Schemas are valid")
    
    # Proceed with generation
    config = ModelConfig(provider="anthropic")
    generator = SyntheticDataGenerator(model_config=config)
    results = generator.generate_for_schemas(schemas=schemas, sample_sizes={"Customer": 5, "Order": 10})
    
except ValueError as e:
    print(f"Schema validation error: {str(e)}")
```

## Debugging Tips

### Enable Debug Logging

Set the logging level to DEBUG for more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
```

### Incremental Generation

For complex schemas, generate one table at a time to isolate issues:

```python
# Generate Customer data first
customer_results = generator.generate_for_schemas(
    schemas={'Customer': schemas['Customer']},
    sample_sizes={'Customer': 5}
)

# Then generate Order data
all_schemas = {'Customer': schemas['Customer'], 'Order': schemas['Order']}
order_results = generator.generate_for_schemas(
    schemas=all_schemas,
    sample_sizes={'Order': 10},
    existing_data={'Customer': customer_results['Customer']}
)
```

### Template Debugging

To debug template rendering issues:

1. Test templates separately with sample data:

```python
import jinja2

# Create environment
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("templates/")
)

# Sample data
sample_data = {
    "customer_name": "Acme Corp",
    "order_id": 1234,
    "items": [
        {"name": "Product A", "quantity": 2, "price": 10.50},
        {"name": "Product B", "quantity": 1, "price": 25.00}
    ]
}

# Render template
template = env.get_template("invoice.html")
try:
    result = template.render(**sample_data)
    print("Template rendered successfully")
except Exception as e:
    print(f"Template error: {str(e)}")
```

2. Use template debugging mode to print undefined variables instead of raising errors:

```python
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("templates/"),
    undefined=jinja2.DebugUndefined
)
```

## Common Troubleshooting Steps

1. **Schema Issues**
   - Verify all field types are valid
   - Check for typos in table and column names
   - Ensure foreign keys reference existing tables

2. **Data Quality Issues**
   - Refine prompts to improve generated content
   - Implement custom generators for complex fields
   - Use custom post-processors to validate and fix data

3. **Performance Issues**
   - Reduce batch sizes for complex schemas
   - Use lower-cost models for initial development
   - Parallelize generation of independent tables

4. **Template Issues**
   - Validate templates with test data before full generation
   - Use conditional checks in templates for optional fields
   - Implement proper error handling in custom generators

## Best Practices for Error Prevention

1. **Start Small**: Begin with simple schemas and gradually add complexity
2. **Test Thoroughly**: Test each schema separately before combining
3. **Use Version Control**: Track changes to schemas and templates
4. **Implement Logging**: Enable detailed logging during development
5. **Validate Early**: Validate schemas before attempting generation
6. **Document Assumptions**: Document expected data formats and relationships
