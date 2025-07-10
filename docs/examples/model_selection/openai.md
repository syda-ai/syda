# Using OpenAI Models with SYDA

This example demonstrates how to use OpenAI's models with SYDA for synthetic data generation. OpenAI offers several models with different capabilities, token limits, and price points.

## Prerequisites

Before running this example, you need to:

1. Install SYDA and its dependencies
2. Set up your OpenAI API key in your environment

You can set the API key in your `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

Or set it as an environment variable before running your script:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Example Code

The following example demonstrates how to configure and use different OpenAI models for synthetic data generation:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define schema for a single table
schemas = {
    'Patient': {
        'patient_id': {'type': 'number', 'description': 'Unique identifier for the patient'},
        'diagnosis_code': {'type': 'text', 'description': 'ICD-10 diagnosis code'},
        'email': {'type': 'email', 'description': 'Patient email address used for communication'},
        'visit_date': {'type': 'date', 'description': 'Date when the patient visited the clinic'},
        'notes': {'type': 'text', 'description': 'Clinical notes for the patient visit'}
    },
    'Claim': {
        'claim_id': {'type': 'number', 'description': 'Unique identifier for the claim'},
        'patient_id': {'type': 'foreign_key', 'description': 'Reference to the patient who made the claim', 'references': {'schema': 'Patient', 'field': 'patient_id'}},
        'diagnosis_code': {'type': 'text', 'description': 'ICD-10 diagnosis code'},
        'email':    {'type': 'email', 'description': 'Patient email address used for communication'},
        'visit_date': {'type': 'date', 'description': 'Date when the patient visited the clinic'},
        'notes': {'type': 'text', 'description': 'Clinical notes for the patient visit'}
    }
}

prompts={
    'Patient': 'Generate realistic synthetic patient records with ICD-10 diagnosis codes, emails, visit dates, and clinical notes.', 
    'Claim': 'Generate realistic synthetic claim records with ICD-10 diagnosis codes, emails, visit dates, and clinical notes.'
}


print("--------------Testing OpenAI GPT-4o----------------")
sample_sizes={'Patient': 15, 'Claim': 15}
model_config = ModelConfig(
    provider="openai",
    model_name="gpt-4o-2024-08-06",
    temperature=0.7,
    max_completion_tokens=16000  # Larger value for more complete responses
)

generator = SyntheticDataGenerator(model_config=model_config)
 # Define output directory
output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "output", 
        "test_openai_models", 
        "gpt-4o"
)
# Generate and save to CSV
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir=output_dir
)
print(f"Data saved to {output_dir}")


print("--------------Testing OpenAI o3----------------")
model_config = ModelConfig(
    provider="openai",
    model_name="o3-2025-04-16",
    max_completion_tokens=100000  # Larger value for more complete responses
)

generator = SyntheticDataGenerator(model_config=model_config)
 # Define output directory
output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "output", 
        "test_openai_models", 
        "o3"
)
sample_sizes={'Patient': 100, 'Claim': 200}
# Generate and save to CSV
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir=output_dir
)
print(f"Data saved to {output_dir}")
```

## OpenAI Model Options

SYDA supports the following OpenAI models:

| Model | Description | Best For | Max Tokens |
|-------|-------------|----------|------------|
| `gpt-4o-2024-08-06` | Versatile model with strong reasoning | General purpose data generation | 16,000 |
| `o3-2025-04-16` | Highest capability model | Complex data with sophisticated relationships | 100,000 |
| `gpt-4-turbo` | Balanced model for most use cases | Medium-sized datasets | 16,000 |

## Key Concepts

### Foreign Key Handling

Foreign keys are crucial for maintaining referential integrity in generated data. In the example above, we're using the `foreign_key` type with explicit references:

```python
'patient_id': {
    'type': 'foreign_key',
    'description': 'Reference to the patient who made the claim',
    'references': {
        'schema': 'Patient', 
        'field': 'patient_id'
    }
}
```

SYDA supports three methods for defining foreign keys:
1. Using the `__foreign_keys__` special section
2. Using field-level references with type and references properties (shown above)
3. Using type-based detection with naming conventions (field name ends with `_id`)

### Model Configuration

The `ModelConfig` class is used to specify which model to use:

```python
model_config = ModelConfig(
    provider="openai",
    model_name="gpt-4o-2024-08-06",
    temperature=0.7,
    max_completion_tokens=16000
)
```

- **provider**: Set to `"openai"` to use OpenAI models
- **model_name**: Specify which OpenAI model to use
- **temperature**: Controls randomness in generation (0.0-1.0)
- **max_completion_tokens**: Maximum number of tokens in the response

### Scaling to Larger Datasets

When generating larger datasets, consider using models with higher token limits:

```python
sample_sizes = {'Patient': 100, 'Claim': 200}
```

The o3 model can handle generating larger datasets in a single request, which is more efficient than making multiple smaller requests.

### Output Directory Structure

The example code creates an organized directory structure for output files:

```
output/
├── test_openai_models/
│   ├── gpt-4o/
│   │   ├── Patient.csv
│   │   └── Claim.csv
│   └── o3/
│       ├── Patient.csv
│       └── Claim.csv
```

## Best Practices

1. **Choose the right model for your task**: 
   - Use gpt-4o for balanced performance and quality
   - Use o3 for complex data structures and larger datasets

2. **Set appropriate token limits**: Different models have different token limits. Make sure to set the `max_completion_tokens` parameter accordingly.

3. **Use detailed prompts**: Include specific instructions in your prompts to get better quality synthetic data.

4. **Monitor API usage**: Keep track of your API usage to manage costs, especially when working with larger datasets.

5. **Handle relationships carefully**: When defining schemas with foreign keys, ensure you specify the relationships correctly for proper data generation.
