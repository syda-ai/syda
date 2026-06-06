"""
Unified LLM client for different providers using pydantic-ai.
Provides a standard interface for creating pydantic-ai Agents with any supported provider.
"""

import os
from typing import Optional, Dict, Any, Union, Type
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from .schemas import ModelConfig


def _build_pydantic_ai_model(model_config: ModelConfig):
    """Build a pydantic-ai model object from ModelConfig."""
    provider = model_config.provider
    model_name = model_config.model_name
    extra = model_config.extra_kwargs or {}

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        api_key = extra.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=api_key) if api_key else AnthropicProvider(),
        )

    elif provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        api_key = extra.get("api_key") or os.environ.get("OPENAI_API_KEY")
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(api_key=api_key) if api_key else OpenAIProvider(),
        )

    elif provider == "gemini":
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.providers.google import GoogleProvider
        api_key = extra.get("api_key") or os.environ.get("GEMINI_API_KEY")
        return GeminiModel(
            model_name,
            provider=GoogleProvider(api_key=api_key) if api_key else GoogleProvider(),
        )

    elif provider == "azureopenai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.azure import AzureProvider
        api_key = extra.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = extra.get("azure_endpoint") or extra.get("api_base")
        api_version = extra.get("api_version", "2024-02-01")
        return OpenAIChatModel(
            model_name,
            provider=AzureProvider(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            ),
        )

    elif provider == "grok":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        base_url = extra.get("base_url", "https://api.x.ai/v1")
        api_key = extra.get("api_key") or os.environ.get("GROK_API_KEY") or "none"
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )

    elif provider == "openai_compatible":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        base_url = extra.get("base_url")
        if not base_url:
            raise ValueError(
                "openai_compatible provider requires 'base_url' in extra_kwargs. "
                "Example: extra_kwargs={'base_url': 'http://localhost:11434/v1', 'api_key': 'ollama'}"
            )
        api_key = extra.get("api_key") or "none"
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


class LLMClient:
    """Unified LLM client wrapping pydantic-ai Agents for structured output generation."""

    def __init__(
        self,
        model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        grok_api_key: Optional[str] = None,
        **kwargs,
    ):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
        if grok_api_key:
            os.environ["GROK_API_KEY"] = grok_api_key

        if model_config is None:
            self.model_config = ModelConfig()
        elif isinstance(model_config, dict):
            self.model_config = ModelConfig(**model_config)
        else:
            self.model_config = model_config

        self._model = _build_pydantic_ai_model(self.model_config)

    def create_agent(self, output_type, system_prompt: str = "", retries: int = 3) -> Agent:
        """Create a pydantic-ai Agent for the given output type."""
        return Agent(self._model, output_type=output_type, system_prompt=system_prompt,
                     retries=retries)

    def get_model_settings(self) -> Optional[ModelSettings]:
        """Build a ModelSettings dict from ModelConfig parameters."""
        settings: Dict[str, Any] = {}
        if self.model_config.temperature is not None:
            settings["temperature"] = self.model_config.temperature
        max_tokens = (
            self.model_config.max_tokens
            or self.model_config.max_completion_tokens
            or self.model_config.max_tokens_to_sample
        )
        if max_tokens is not None:
            settings["max_tokens"] = max_tokens
        if self.model_config.top_p is not None:
            settings["top_p"] = self.model_config.top_p
        return ModelSettings(**settings) if settings else None

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Legacy helper — returns model name dict for backward compatibility."""
        return {"model": self.model_config.model_name}


def create_llm_client(
    model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    grok_api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create an LLMClient with the specified configuration."""
    return LLMClient(
        model_config=model_config,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        grok_api_key=grok_api_key,
        **kwargs,
    )
