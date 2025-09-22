---
title: AI Model Configuration & Selection | Syda Deep Dive
description: Configure and select AI models (OpenAI, Anthropic, Gemini, Azure OpenAI) for synthetic data generation with Syda - API keys, model parameters, and performance optimization.
keywords:
  - AI model configuration
  - LLM model selection
  - OpenAI model setup
  - Anthropic model setup
  - Gemini model setup
  - Azure OpenAI model setup
  - API key configuration
  - model parameters
  - extra_kwargs configuration
---

# Model Selection and Configuration

SYDA supports multiple large language model (LLM) providers, allowing you to choose the model that best fits your needs in terms of capabilities, cost, and performance.

## API Keys

Before using any LLM provider, you must set the appropriate API keys as environment variables:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY=your_anthropic_key

# For OpenAI
export OPENAI_API_KEY=your_openai_key

# For Azure OpenAI
export AZURE_OPENAI_API_KEY=your_azure_openai_key

# For Gemini
export GEMINI_API_KEY=your_gemini_key
```

Alternatively, you can create a `.env` file in your project root:

```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_KEY=your_azure_openai_key
GEMINI_API_KEY=your_gemini_key
```

Refer to the [Quickstart Guide](../quickstart.md) for more details on environment setup.

## Basic Configuration

The `ModelConfig` class is used to specify which LLM provider and model you want to use:

```python
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Basic configuration with default parameters
config = ModelConfig(
    provider="anthropic",  # Choose provider: 'anthropic', 'openai', 'azureopenai', 'gemini'
    model_name="claude-3-5-haiku-20241022",  # Specify model name
    temperature=0.7,  # Control randomness (0.0-1.0)
    max_tokens=8000   # Maximum number of tokens to generate
)

# Initialize generator with this configuration
generator = SyntheticDataGenerator(model_config=config)
```

## Using Different Model Providers

SYDA supports multiple LLM providers:

### Anthropic Claude Models

Claude is the default model provider for SYDA, offering strong performance for data generation tasks:

```python
# Using Anthropic Claude
config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",  # Default model
    temperature=0.5,  # Control randomness (0.0-1.0)
    max_tokens=8000   # Maximum number of tokens to generate
)
```

Available Claude models include:

For the latest information about available Claude models and their capabilities, refer to [Anthropic's Claude documentation](https://docs.anthropic.com/en/docs/about-claude/models/overview).

### OpenAI Models

SYDA also supports OpenAI's GPT models:

```python
# Using OpenAI GPT
config = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo",
    temperature=0.7,
    max_tokens=4000
)
```

For the latest information about available OpenAI models and their capabilities, refer to [OpenAI's models documentation](https://platform.openai.com/docs/models).

### Azure OpenAI Models

SYDA supports Azure OpenAI for enterprise deployments:

```python
# Using Azure OpenAI
config = ModelConfig(
    provider="azureopenai",
    model_name="gpt-4o",  # Your deployment name
    temperature=0.7,
    max_tokens=4000,
    extra_kwargs={
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2024-02-15-preview"
    }
)
```

See the [Azure OpenAI Example](../examples/model_selection/azureopenai.md) for detailed setup instructions.

### Gemini Models

SYDA also supports Google's Gemini models:

```python
# Using Gemini
config = ModelConfig(
    provider="gemini",
    model_name="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=4000
)
```

## Model Parameters

You can fine-tune model behavior with these parameters:

| Parameter | Description | Range | Default |
|-----------|-------------|-------|--------|
| `temperature` | Controls randomness in generation | 0.0-1.0 | None |
| `max_tokens` | Maximum tokens to generate | Integer | None |
| `max_completion_tokens` | Maximum completion tokens to generate for latest openai models | Integer | None |

## Advanced Configuration with extra_kwargs

The `extra_kwargs` parameter allows you to pass provider-specific configuration options directly to the underlying LLM client. This is particularly useful for:

- Custom endpoints and base URLs
- AI gateway integration (LiteLLM, Portkey, Kong, etc.)
- Timeout and retry configurations  
- HTTP client customization
- Azure OpenAI specific parameters
- Authentication headers and tokens
- Any other provider-specific settings

### General Usage

```python
config = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo",
    temperature=0.7,
    max_tokens=4000,
    extra_kwargs={
        "base_url": "https://custom-openai-proxy.com/v1",
        "timeout": 60,
        "max_retries": 3
    }
)
```

### Provider-Specific Examples

#### OpenAI extra_kwargs

```python
config = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo",
    extra_kwargs={
        "base_url": "https://custom-openai-endpoint.com/v1",
        "timeout": 120,
        "max_retries": 5,
        "default_headers": {"Custom-Header": "value"}
    }
)
```

#### Azure OpenAI extra_kwargs (Required)

```python
config = ModelConfig(
    provider="azureopenai",
    model_name="gpt-4o",  # Your deployment name
    extra_kwargs={
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",  # Required
        "api_version": "2024-02-15-preview",  # Required
        "azure_deployment": "custom-deployment-name",  # Optional if different from model_name
        "timeout": 60,
        "max_retries": 2
    }
)
```

#### Anthropic extra_kwargs

```python
config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    extra_kwargs={
        "base_url": "https://custom-anthropic-endpoint.com",
        "timeout": 90,
        "max_retries": 3,
        "default_headers": {"Custom-Header": "value"}
    }
)
```

#### Gemini extra_kwargs

```python
config = ModelConfig(
    provider="gemini",
    model_name="gemini-1.5-flash",
    extra_kwargs={
        "transport": "grpc",  # or "rest"
        "client_options": {"api_endpoint": "custom-endpoint.googleapis.com"}
    }
)
```

### Common extra_kwargs Parameters

| Parameter | Description | Applicable Providers |
|-----------|-------------|---------------------|
| `timeout` | Request timeout in seconds | All providers |
| `max_retries` | Maximum retry attempts | All providers |
| `base_url` | Custom API endpoint (for gateways and proxies) | OpenAI, Anthropic |
| `azure_endpoint` | Azure OpenAI endpoint URL | Azure OpenAI (Required) |
| `api_version` | Azure API version | Azure OpenAI (Required) |
| `azure_deployment` | Azure deployment name | Azure OpenAI (Optional) |
| `default_headers` | Custom HTTP headers (for gateway authentication) | OpenAI, Anthropic |
| `api_key` | Custom API key (for gateway authentication) | All providers |

### AI Gateway Integration

The `extra_kwargs` parameter is particularly useful for integrating with AI gateways and proxy services that provide unified access to multiple LLM providers:

#### LiteLLM Gateway

```python
config = ModelConfig(
    provider="openai",  # Use OpenAI-compatible format
    model_name="gpt-4-turbo",
    extra_kwargs={
        "base_url": "http://localhost:4000",  # LiteLLM proxy endpoint
        "api_key": "your-litellm-key",
        "default_headers": {
            "User-Agent": "syda-client/1.0"
        }
    }
)
```

#### Portkey Gateway

```python
config = ModelConfig(
    provider="openai",  # Portkey uses OpenAI-compatible API
    model_name="gpt-4-turbo",
    extra_kwargs={
        "base_url": "https://api.portkey.ai/v1",
        "default_headers": {
            "x-portkey-api-key": "your-portkey-api-key",
            "x-portkey-provider": "openai",
            "x-portkey-trace-id": "syda-session"
        }
    }
)
```

#### Kong AI Gateway

```python
config = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo", 
    extra_kwargs={
        "base_url": "https://your-kong-gateway.com/ai/v1",
        "default_headers": {
            "Authorization": "Bearer your-kong-token",
            "Kong-Route-Id": "openai-route"
        },
        "timeout": 120
    }
)
```

#### Custom AI Gateway

```python
config = ModelConfig(
    provider="openai",  # Most gateways use OpenAI-compatible format
    model_name="your-custom-model",
    extra_kwargs={
        "base_url": "https://your-custom-gateway.com/v1",
        "api_key": "your-gateway-token",
        "default_headers": {
            "X-Gateway-Version": "v2",
            "X-Client": "syda",
            "X-Request-Source": "synthetic-data-generation"
        },
        "timeout": 180,
        "max_retries": 2
    }
)
```

### When to Use extra_kwargs

- **Enterprise Deployments**: Custom endpoints, proxy servers, or private cloud deployments
- **Azure OpenAI**: Required for all Azure OpenAI configurations
- **AI Gateway Integration**: Connect to LiteLLM, Portkey, Kong, or custom AI gateways
- **Performance Tuning**: Custom timeout and retry settings
- **Authentication**: Custom headers or authentication methods
- **Development/Testing**: Local proxy servers or mock endpoints
- **Cost Management**: Route through gateways that provide usage tracking and cost optimization
- **Multi-Provider Access**: Use gateways that provide unified access to multiple LLM providers



## Advanced: Direct Access to LLM Client

For advanced use cases, you can access the underlying LLM client directly:

```python
from syda import SyntheticDataGenerator, ModelConfig

config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)

# Access the underlying client
llm_client = generator.llm_client

# Use the client directly (provider-specific)
if config.provider == "anthropic":
    response = llm_client.messages.create(
        model=config.model_name,
        max_tokens=1000,
        messages=[{"role": "user", "content": "Generate a list of book titles about AI"}]
    )
    
    print(response.content[0].text)
```

This gives you direct access to provider-specific features while still using SYDA for schema management.

## Best Practices

1. **Start with Default Models**: Begin with `claude-3-5-haiku-20241022` (Anthropic) or `gpt-4-turbo` (OpenAI)
2. **Adjust Temperature**: Lower for more consistent results, higher for more variety
3. **Consider Cost vs. Quality**: Higher-end models provide better quality but at higher cost
4. **Test Different Models**: Compare results from different models for your specific use case
5. **Handle API Rate Limits**: Implement appropriate backoff strategies for large generations
6. **Select Highest Output Token Model for Higher Sample Sizes**: For larger sample sizes, use models with higher `max_tokens`
7. **Use extra_kwargs for Customization**: Leverage `extra_kwargs` for enterprise deployments and custom configurations
8. **Secure API Keys**: Never hardcode API keys; always use environment variables or secure key management



## Examples

Explore these model-specific examples to see configuration in action:
- [Anthropic Claude Example](../examples/model_selection/anthropic.md)
- [OpenAI GPT Example](../examples/model_selection/openai.md) 
- [Azure OpenAI Example](../examples/model_selection/azureopenai.md)
- [Gemini Example](../examples/model_selection/gemini.md) 