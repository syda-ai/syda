# Synthetic Data Generation Library

A Python-based open-source library for generating synthetic data with AI while preserving referential integrity.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
  - [Structured Data Generation](#structured-data-generation)
  - [SQLAlchemy Model Integration](#sqlalchemy-model-integration)
  - [Handling Foreign Key Relationships](#handling-foreign-key-relationships)
  - [Automatic Management of Multiple Related Models](#automatic-management-of-multiple-related-models)
  - [Complete CRM Example](#complete-crm-example)
  - [Metadata Enhancement Benefits](#metadata-enhancement-benefits)
  - [Model Selection and Configuration](#model-selection-and-configuration)
  - [Output Options](#output-options)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Features

### Features

- **Synthetic Data Generation**:
  - Statistical data generation
  - Pattern-based generation
  - Data distribution preservation
  - Synthetic data from various sources

## Core Module Usage

### Installation

Install the package using pip:

```bash
pip install syda
```

### Basic Usage

#### Synthetic Data Generation

##### Model Selection and Configuration

The library allows you to customize which AI model to use for generating data and configure model-specific parameters. Additionally, you can configure proxy settings for the API requests.

```python
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Configure with default settings (OpenAI GPT-4)
generator = SyntheticDataGenerator()

# Configure with specific model name
generator = SyntheticDataGenerator(
    model_config={"model_name": "gpt-3.5-turbo"}
)

# Use Anthropic Claude model with specific settings
claude_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-opus-20240229",
    temperature=0.5,
    max_tokens=2000
)
generator = SyntheticDataGenerator(model_config=claude_config)

# Provide custom API keys
generator = SyntheticDataGenerator(
    model_config={"model_name": "gpt-4-turbo"},
    openai_api_key="your-openai-api-key"
)

# Using with company AI proxy
from syda.schemas import ProxyConfig

generator = SyntheticDataGenerator(
    model_config={
        "model_name": "gpt-4-turbo",
        "proxy": {
            "base_url": "https://ai-proxy.company.com/v1",
            "headers": {"X-Company-Auth": "your-internal-token"}
        }
    }
)

# Or with the ModelConfig class for more control
proxy_config = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo",
    proxy=ProxyConfig(
        base_url="https://ai-proxy.company.com/v1",
        headers={"X-Company-Auth": "your-internal-token"},
        params={"team": "data-science"}
    )
)
```

You can customize various model parameters:

1. **Common Parameters**:
   - `provider`: The AI provider to use - "openai" or "anthropic" (default: "openai")
   - `model_name`: The specific model to use (default: "gpt-4")
   - `temperature`: Controls randomness of outputs (0.0-2.0, default: 0.7)
   - `max_tokens`: Maximum tokens to generate
   - `top_p`: Nucleus sampling parameter (0.0-1.0)

2. **OpenAI Specific**:
   - `seed`: Random seed for reproducibility
   - `response_format`: Format specification for responses

3. **Anthropic Specific**:
   - `top_k`: Top K sampling parameter
   - `max_tokens_to_sample`: Maximum tokens to generate (Anthropic specific)

4. **Proxy Configuration** ⚠️:
   - `proxy`: Configure API requests to use your company's AI proxy service
     - `base_url`: Base URL for the proxy service
     - `headers`: Additional HTTP headers for authentication and metadata
     - `params`: Query parameters to include in requests
     - `path_format`: Custom path formatting (if needed by your proxy)
   
   > **⚠️ Warning**: The proxy configuration feature is experimental and has not been thoroughly tested with actual enterprise proxy setups. You may need to adjust the implementation to work with your specific company proxy. Please report any issues you encounter.

Example with advanced configuration:

```python
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig, ProxyConfig

# Create a model configuration with specific settings
config = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo",
    temperature=0.9,
    top_p=0.95,
    seed=42,  # For reproducible results
    max_tokens=1000,
    proxy=ProxyConfig(
        base_url="https://ai-proxy.company.com/v1",
        headers={
            "X-Company-Auth": "internal-token-123",
            "X-Project-ID": "synthetic-data"
        },
        params={"team": "data-science"},
        path_format="/proxy/{provider}/completions"
    )
)

# Initialize generator with the configuration
generator = SyntheticDataGenerator(model_config=config)

# Generate data using the configured model
data = generator.generate_data(
    schema=my_schema,
    prompt="Generate realistic customer data for a SaaS company",
    sample_size=10
)
```

##### Basic Usage

You can generate synthetic data and write it directly to a CSV file using the `output_path` argument:
