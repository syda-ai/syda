---
title: OpenAI-Compatible Providers | Syda Examples
description: Use Syda with any OpenAI-compatible API — Ollama, Groq, Together AI, Fireworks, DeepSeek, Mistral, LM Studio, vLLM, and more.
keywords:
  - Ollama synthetic data
  - Groq synthetic data
  - local LLM synthetic data
  - openai compatible provider
  - self-hosted LLM
---

# OpenAI-Compatible Providers

Syda's `openai_compatible` provider lets you connect to **any server that speaks the OpenAI API** — local models via Ollama, cloud providers like Groq and Together AI, or self-hosted inference engines like vLLM and LM Studio.

## Installation

```bash
pip install syda
```

No extra dependencies needed — `openai` is already a Syda dependency.

## Ollama (Local Models)

Run models entirely on your own machine — no API key, no cloud, no cost.

```bash
# Install and start Ollama
brew install ollama        # macOS
brew services start ollama

# Pull a model
ollama pull llama3
ollama pull gpt-oss:20b
```

```python
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv

load_dotenv()

generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="openai_compatible",
        model_name="llama3",
        temperature=0.7,
        max_tokens=2048,
        extra_kwargs={
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",  # Ollama doesn't validate the key — any string works
        }
    )
)

results = generator.generate_for_schemas(
    schemas={
        "users": {
            "name": {"type": "string"},
            "age":  {"type": "integer"},
            "city": {"type": "string"},
        }
    },
    sample_sizes={"users": 10},
    prompts={"users": "Generate realistic user records."},
    output_dir="output"
)
```

## Groq (Fast Cloud Inference)

Groq offers free-tier access with very fast inference speeds.

```python
import os

generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="openai_compatible",
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=4096,
        extra_kwargs={
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.environ["GROQ_API_KEY"],
        }
    )
)
```

## Together AI

```python
generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="openai_compatible",
        model_name="meta-llama/Llama-3-8b-chat-hf",
        extra_kwargs={
            "base_url": "https://api.together.xyz/v1",
            "api_key": os.environ["TOGETHER_API_KEY"],
        }
    )
)
```

## DeepSeek

```python
generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="openai_compatible",
        model_name="deepseek-chat",
        extra_kwargs={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": os.environ["DEEPSEEK_API_KEY"],
        }
    )
)
```

## Mistral

```python
generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="openai_compatible",
        model_name="mistral-small-latest",
        extra_kwargs={
            "base_url": "https://api.mistral.ai/v1",
            "api_key": os.environ["MISTRAL_API_KEY"],
        }
    )
)
```

## `response_mode` — Handling Different Response Formats

Different models return structured data in different ways. Use `response_mode` to tell Syda how to parse the response:

| Value | When to use |
|---|---|
| `"markdown"` | **Default.** Model wraps JSON in ` ```json``` ` fences (most local models) |
| `"tools"` | Model supports tool calls natively — best structured output quality |
| `"json"` | Model returns clean JSON with no markdown wrapping |

```python
extra_kwargs={
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "response_mode": "tools",   # use tool calls for models that support it
}
```

!!! tip
    If you're unsure which mode to use, leave it unset — `"markdown"` is the safe default that works with most local models.

## Supported Providers

| Provider | `base_url` | Notes |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | Local, free, no key needed |
| Groq | `https://api.groq.com/openai/v1` | Free tier available |
| Together AI | `https://api.together.xyz/v1` | Many open-source models |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` | Fast inference |
| DeepSeek | `https://api.deepseek.com/v1` | Cost-effective |
| Mistral | `https://api.mistral.ai/v1` | European provider |
| LM Studio | `http://localhost:1234/v1` | Local GUI app |
| vLLM | `http://localhost:8000/v1` | Self-hosted, high throughput |
| Perplexity | `https://api.perplexity.ai` | Search-augmented models |

Any server implementing the OpenAI `/v1/chat/completions` endpoint will work.
