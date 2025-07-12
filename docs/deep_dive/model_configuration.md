# Model Selection and Configuration

SYDA supports multiple large language model (LLM) providers, allowing you to choose the model that best fits your needs in terms of capabilities, cost, and performance.

## API Keys

Before using any LLM provider, you must set the appropriate API keys as environment variables:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY=your_anthropic_key

# For OpenAI
export OPENAI_API_KEY=your_openai_key
```

Alternatively, you can create a `.env` file in your project root:

```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

Refer to the [Quickstart Guide](../quickstart.md) for more details on environment setup.

## Basic Configuration

The `ModelConfig` class is used to specify which LLM provider and model you want to use:

```python
from syda import SyntheticDataGenerator, ModelConfig

# Basic configuration with default parameters
config = ModelConfig(
    provider="anthropic",  # Choose provider: 'anthropic' or 'openai'
    model_name="claude-3-5-haiku-20241022",  # Specify model name
    temperature=0.7,  # Control randomness (0.0-1.0)
    max_tokens=8000   # Maximum number of tokens to generate
)

# Initialize generator with this configuration
generator = SyntheticDataGenerator(model_config=config)
```

## Using Different Model Providers

SYDA currently supports two main LLM providers:

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

## Model Parameters

You can fine-tune model behavior with these parameters:

| Parameter | Description | Range | Default |
|-----------|-------------|-------|--------|
| `temperature` | Controls randomness in generation | 0.0-1.0 | None |
| `max_tokens` | Maximum tokens to generate | Integer | None |
| `max_completion_tokens` | Maximum completion tokens to generate for latest openai models | Integer | None |



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



## Examples

Explore  [Anthropic Claude Example](../examples/model_selection/anthropic.md) and [Openai Gpt Example](../examples/model_selection/openai.md) to see model configuration in action. 