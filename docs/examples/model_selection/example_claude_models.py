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
sample_sizes={'Patient': 15, 'Claim': 15}

print("--------------Testing Claude Haiku----------------")
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192  # Larger value for more complete responses
)

generator = SyntheticDataGenerator(model_config=model_config)
 # Define output directory
output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "output", 
        "test_claude_models", 
        "haiku-3-5"
)
# Generate and save to CSV
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir=output_dir
)
print(f"Data saved to {output_dir}")


print("--------------Testing Claude Sonnet----------------")
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=64000  # Larger value for more complete responses
)

generator = SyntheticDataGenerator(model_config=model_config)
 # Define output directory
output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "output", 
        "test_claude_models", 
        "sonnet-4"
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

print("--------------Testing Claude Opus----------------")
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-opus-4-20250514",
    temperature=0.7,
    max_tokens=32000  # Larger value for more complete responses
)

generator = SyntheticDataGenerator(model_config=model_config)
 # Define output directory
output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "output", 
        "test_claude_models", 
        "opus-4"
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