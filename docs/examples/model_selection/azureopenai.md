---
title: Azure OpenAI Models for Synthetic Data | Syda Examples
description: Learn how to use Azure OpenAI models (GPT-4o, GPT-4o-mini) with Syda for AI-powered synthetic data generation - Azure configuration, deployment setup, and best practices.
keywords:
  - Azure OpenAI synthetic data
  - GPT-4o Azure deployment
  - Azure OpenAI data generation
  - Syda Azure integration
  - Azure OpenAI configuration
  - Enterprise AI data generation
---

# Using Azure OpenAI Models with SYDA

> Source code: [examples/model_selection/example_azureopenai_models.py](https://github.com/syda-ai/syda/blob/main/examples/model_selection/example_azureopenai_models.py)

This example demonstrates how to use Azure OpenAI models with SYDA for synthetic data generation. Azure OpenAI provides enterprise-grade access to OpenAI models with enhanced security, compliance, and regional deployment options.

## Prerequisites

Before running this example, you need to:

1. **Azure OpenAI Resource**: Deploy an Azure OpenAI resource in your Azure subscription
2. **Model Deployments**: Create model deployments in Azure OpenAI Studio
3. **API Access**: Obtain your API key and endpoint URL from the Azure portal
4. **SYDA Installation**: Install SYDA and its dependencies

### Azure OpenAI Setup

1. **Create Azure OpenAI Resource**:
   - Go to Azure Portal → Create Resource → Azure OpenAI
   - Choose your subscription, resource group, and region
   - Note your endpoint URL (e.g., `https://your-resource-name.openai.azure.com/`)

2. **Deploy Models**:
   - Navigate to Azure OpenAI Studio
   - Go to "Deployments" → "Create new deployment"
   - Deploy models like `gpt-4o`, `gpt-4o-mini`, etc.
   - Note your deployment names

3. **Get API Key**:
   - In Azure Portal, go to your Azure OpenAI resource
   - Navigate to "Keys and Endpoint"
   - Copy one of the API keys

## Environment Configuration

Set your Azure OpenAI API key in your `.env` file:

```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
```

Or set it as an environment variable:

```bash
export AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
```

## Example Code

The following example demonstrates how to configure and use Azure OpenAI models for synthetic data generation:

```python
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

# Azure OpenAI Configuration
model_config_gpt4o = ModelConfig(
    provider="azureopenai",
    model_name="gpt-4o",  # This should match your deployment name in Azure
    temperature=0.7,
    max_tokens=4000,
    extra_kwargs={
        # Required Azure OpenAI parameters
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",  # Replace with your endpoint
        "api_version": "2024-02-15-preview",  # Use the latest API version
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
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir=output_dir
)
print(f"✅ GPT-4o data saved to {output_dir}")
print(f"Generated {len(results['Patient'])} patients and {len(results['Appointment'])} appointments")
```



## Azure OpenAI Configuration

### Required Parameters

Azure OpenAI requires specific configuration parameters passed through the `extra_kwargs` field:

```python
extra_kwargs={
    "azure_endpoint": "https://your-resource-name.openai.azure.com/",  # Your Azure endpoint
    "api_version": "2024-02-15-preview",  # API version
}
```

### Optional Parameters

You can also include optional parameters for advanced configuration:

```python
extra_kwargs={
    "azure_endpoint": "https://your-resource-name.openai.azure.com/",
    "api_version": "2024-02-15-preview",
}
```



## Key Concepts

### Model Configuration for Azure OpenAI

The `ModelConfig` class requires specific settings for Azure OpenAI:

```python
model_config = ModelConfig(
    provider="azureopenai",  # Use Azure OpenAI provider
    model_name="gpt-4o",     # Must match your deployment name
    temperature=0.7,
    max_tokens=4000,
    extra_kwargs={
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2024-02-15-preview"
    }
)
```

**Important**: The `model_name` should match the deployment name you created in Azure OpenAI Studio.

### Foreign Key Relationships

The example demonstrates proper foreign key configuration for maintaining referential integrity:

```python
'patient_id': {
    'type': 'foreign_key', 
    'description': 'Reference to the patient', 
    'references': {'schema': 'Patient', 'field': 'patient_id'}
}
```

This ensures that all appointment records reference valid patient IDs from the Patient table.

### API Version Management

Azure OpenAI uses API versions for feature compatibility. Use the latest stable version:

- **2024-02-15-preview**: Latest preview with newest features
- **2023-12-01-preview**: Stable version for production use
- **2023-05-15**: Older stable version

## Best Practices

### 1. Security and Compliance

- **Use Managed Identity**: Configure Azure Managed Identity for secure access without API keys
- **Private Endpoints**: Deploy Azure OpenAI in your VNet using private endpoints
- **Access Controls**: Use Azure RBAC to control access to your Azure OpenAI resource

### 2. Performance Optimization

- **Choose Appropriate Regions**: Deploy in regions close to your users for better latency
- **Model Selection**: 
  - Use `gpt-4o-mini` for cost-effective generation
  - Use reasoning models for highest quality output

### 3. Cost Management

- **Monitor Usage**: Use Azure Cost Management to track API usage and costs
- **Set Quotas**: Configure quotas in Azure OpenAI Studio to control spending
- **Optimize Token Usage**: Set appropriate `max_tokens` limits based on your schema complexity



## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Verify `AZURE_OPENAI_API_KEY` is set correctly
   - Check API key permissions in Azure portal

2. **Deployment Not Found**:
   - Ensure `model_name` matches your deployment name exactly
   - Verify deployment is active in Azure OpenAI Studio

3. **API Version Errors**:
   - Use supported API versions (check Azure OpenAI documentation)
   - Update to latest stable version if issues persist

4. **Network Connectivity**:
   - Check if private endpoints are configured correctly
   - Verify firewall rules allow access to Azure OpenAI

### Getting Help

- **Azure Support**: Use Azure Support for infrastructure and deployment issues
- **SYDA Community**: Open Github issues, https://github.com/syda-ai/syda/issues
- **Documentation**: Refer to Azure OpenAI documentation for service-specific guidance

## Output Directory Structure

The example creates an organized directory structure for output files:

```
output/
└── test_azureopenai_models/
    └── gpt-4o/
        ├── Patient.csv
        └── Appointment.csv
```

