"""
Azure OpenAI Model Configuration Example for Syda

This example demonstrates how to configure and use Azure OpenAI models with Syda.
Azure OpenAI requires additional configuration parameters that are passed via extra_kwargs.

Prerequisites:
1. Azure OpenAI resource deployed in Azure
2. Model deployments created in Azure OpenAI Studio
3. API key and endpoint URL from your Azure OpenAI resource

Environment Variables:
- AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
- Or pass the API key directly to the SyntheticDataGenerator constructor

Required extra_kwargs for Azure OpenAI:
- azure_endpoint: Your Azure OpenAI endpoint URL
- api_version: The API version to use (e.g., "2024-02-15-preview")
- azure_deployment: Optional, can be set here or use model_name as deployment name
"""

from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define schema for healthcare data
schemas = {
    'Patient': {
        'patient_id': {'type': 'number', 'description': 'Unique identifier for the patient'},
        'first_name': {'type': 'text', 'description': 'Patient first name'},
        'last_name': {'type': 'text', 'description': 'Patient last name'},
        'email': {'type': 'email', 'description': 'Patient email address'},
        'phone': {'type': 'phone', 'description': 'Patient contact phone number'},
        'date_of_birth': {'type': 'date', 'description': 'Patient date of birth'},
        'diagnosis_code': {'type': 'text', 'description': 'Primary ICD-10 diagnosis code'},
        'visit_date': {'type': 'date', 'description': 'Date of most recent clinic visit'},
        'notes': {'type': 'text', 'description': 'Clinical notes and observations'}
    },
    'Appointment': {
        'appointment_id': {'type': 'number', 'description': 'Unique identifier for the appointment'},
        'patient_id': {
            'type': 'foreign_key', 
            'description': 'Reference to the patient', 
            'references': {'schema': 'Patient', 'field': 'patient_id'}
        },
        'appointment_date': {'type': 'datetime', 'description': 'Scheduled appointment date and time'},
        'appointment_type': {'type': 'text', 'description': 'Type of appointment (consultation, follow-up, etc.)'},
        'duration_minutes': {'type': 'number', 'description': 'Appointment duration in minutes'},
        'status': {'type': 'text', 'description': 'Appointment status (scheduled, completed, cancelled)'},
        'provider_name': {'type': 'text', 'description': 'Name of the healthcare provider'}
    }
}

prompts = {
    'Patient': 'Generate realistic synthetic patient records for a general practice clinic with diverse demographics, common diagnoses, and clinical notes.',
    'Appointment': 'Generate realistic appointment records with various appointment types, realistic scheduling patterns, and appropriate durations.'
}

# Example 1: Azure OpenAI with GPT-4o
print("--------------Testing Azure OpenAI GPT-4o----------------")

# Configuration for Azure OpenAI GPT-4o
model_config_gpt4o = ModelConfig(
    provider="azureopenai",
    model_name="gpt-4o",  # This should match your deployment name in Azure
    temperature=0.7,
    max_tokens=4000,
    extra_kwargs={
        # Required Azure OpenAI parameters
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",  # Replace with your endpoint
        "api_version": "2024-02-15-preview",  # Use the latest API version
        
        # Optional: If your deployment name differs from model_name
        # "azure_deployment": "your-gpt4o-deployment-name",
        
        # Optional: Additional configuration
        # "timeout": 60,
        # "max_retries": 3
    }
)

# Initialize generator with Azure OpenAI
generator = SyntheticDataGenerator(
    model_config=model_config_gpt4o,
    # You can pass the API key directly or set AZURE_OPENAI_API_KEY environment variable
    # openai_api_key="your-azure-openai-api-key"
)

# Define output directory
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "output", 
    "test_azureopenai_models", 
    "gpt-4o"
)

sample_sizes = {'Patient': 20, 'Appointment': 30}

# Generate and save to CSV
try:
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=output_dir
    )
    print(f"✅ GPT-4o data saved to {output_dir}")
    print(f"Generated {len(results['Patient'])} patients and {len(results['Appointment'])} appointments")
except Exception as e:
    print(f"❌ Error with GPT-4o: {str(e)}")
    print("Please check your Azure OpenAI configuration and ensure:")
    print("1. Your azure_endpoint is correct")
    print("2. Your API key is valid")
    print("3. Your deployment name matches the model_name")
    print("4. Your API version is supported")


# Example 2: Azure OpenAI with GPT-4o-mini (more cost-effective)
print("\n--------------Testing Azure OpenAI GPT-4o-mini----------------")

model_config_gpt4o_mini = ModelConfig(
    provider="azureopenai",
    model_name="gpt-4o-mini",  # This should match your deployment name
    temperature=0.8,
    max_tokens=2000,
    extra_kwargs={
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",  # Replace with your endpoint
        "api_version": "2024-02-15-preview",
        
        # Example of using different deployment name
        # "azure_deployment": "gpt-4o-mini-deployment",
    }
)

generator_mini = SyntheticDataGenerator(model_config=model_config_gpt4o_mini)

output_dir_mini = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "output", 
    "test_azureopenai_models", 
    "gpt-4o-mini"
)

sample_sizes_mini = {'Patient': 50, 'Appointment': 75}

try:
    results_mini = generator_mini.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes_mini,
        output_dir=output_dir_mini
    )
    print(f"✅ GPT-4o-mini data saved to {output_dir_mini}")
    print(f"Generated {len(results_mini['Patient'])} patients and {len(results_mini['Appointment'])} appointments")
except Exception as e:
    print(f"❌ Error with GPT-4o-mini: {str(e)}")


# Example 3: Azure OpenAI with custom configuration and streaming
print("\n--------------Testing Azure OpenAI with Streaming----------------")

model_config_streaming = ModelConfig(
    provider="azureopenai",
    model_name="gpt-4o",
    temperature=0.6,
    max_tokens=8000,
    stream=True,  # Enable streaming for large datasets
    extra_kwargs={
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2024-02-15-preview",
        "timeout": 120,  # Longer timeout for streaming
        "max_retries": 2
    }
)

generator_streaming = SyntheticDataGenerator(model_config=model_config_streaming)

output_dir_streaming = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "output", 
    "test_azureopenai_models", 
    "streaming"
)

# Larger sample sizes to demonstrate streaming
sample_sizes_streaming = {'Patient': 100, 'Appointment': 150}

try:
    results_streaming = generator_streaming.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes_streaming,
        output_dir=output_dir_streaming
    )
    print(f"✅ Streaming data saved to {output_dir_streaming}")
    print(f"Generated {len(results_streaming['Patient'])} patients and {len(results_streaming['Appointment'])} appointments")
except Exception as e:
    print(f"❌ Error with streaming: {str(e)}")
