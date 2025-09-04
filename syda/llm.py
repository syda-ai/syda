"""
Unified LLM client initialization for different providers.
Provides a standard interface for creating LLM clients with instructor integration.
"""

import os
import instructor
import openai
from typing import Optional, Dict, Any, Union, List
from .schemas import ModelConfig, ProxyConfig

try:
    from deepseek import Client as DeepSeekClient  # Use the actual client class name
except ImportError:
    DeepSeekClient = None

class LLMClient:
    """
    A unified client for LLM providers with instructor integration.
    Supports OpenAI, Anthropic, and other providers via the instructor library.
    """

    def __init__(
        self,
        model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,  # Add DeepSeek API key
        **kwargs
    ):
        """
        Initialize an LLM client for the specified model provider.

        Args:
            model_config: Configuration for the LLM model to use
            openai_api_key: Optional API key for OpenAI
            anthropic_api_key: Optional API key for Anthropic
            deepseek_api_key: Optional API key for DeepSeek
            **kwargs: Additional keyword arguments to pass to the client
        """
        # Set up API keys from arguments or environment variables
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")  # Add DeepSeek API key

        # Set up model configuration
        if model_config is None:
            self.model_config = ModelConfig()
        elif isinstance(model_config, dict):
            self.model_config = ModelConfig(**model_config)
        else:
            self.model_config = model_config

        # Store additional kwargs
        self.kwargs = kwargs

        # Initialize the client
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """
        Initialize and return the appropriate LLM client based on the model configuration.

        Returns:
            An instructor-patched client for the specified provider
        """
        # Build provider string
        provider = self.model_config.provider
        model_name = self.model_config.model_name

        # Initialize based on provider
        if provider == "openai":
            # Set up environment variable for OpenAI instead of passing directly
            if self.openai_api_key:
                os.environ["OPENAI_API_KEY"] = self.openai_api_key

            # Get proxy configuration
            proxy_kwargs = {}
            if self.model_config.proxy:
                proxy_kwargs = self.model_config.proxy.get_proxy_kwargs()

            # Initialize raw client
            raw_client = openai.OpenAI(**proxy_kwargs)

            # Patch with instructor and return
            return instructor.from_openai(raw_client)

        elif provider == "anthropic":
            # Set up environment variable for Anthropic instead of passing directly
            if self.anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

            # Initialize with from_anthropic
            try:
                # Try using from_anthropic if available
                from anthropic import Anthropic

                # Get proxy configuration
                proxy_kwargs = {}
                if self.model_config.proxy:
                    proxy_kwargs = self.model_config.proxy.get_proxy_kwargs()

                # Create raw client
                raw_client = Anthropic(**proxy_kwargs)

                # Patch with instructor
                return instructor.patch(raw_client)
            except Exception as e:
                print(f"Warning: Could not use from_anthropic: {e}. Trying from_provider instead.")

                # Fall back to from_provider
                try:
                    # Create client using from_provider
                    return instructor.from_provider(f"{provider}/{model_name}")
                except Exception as inner_e:
                    # All methods failed
                    error_msg = f"Failed to initialize Anthropic client: {e}, {inner_e}"
                    raise ValueError(error_msg)
        elif provider == "deepseek":
            # Set up environment variable for DeepSeek instead of passing directly
            if self.deepseek_api_key:
                os.environ["DEEPSEEK_API_KEY"] = self.deepseek_api_key

            # Initialize with DeepSeek client if available
            try:
                if DeepSeekClient is None:
                    raise ImportError("deepseek package not installed")
                # Get proxy configuration
                proxy_kwargs = {}
                if self.model_config.proxy:
                    proxy_kwargs = self.model_config.proxy.get_proxy_kwargs()
                # Create raw client
                raw_client = DeepSeekClient(**proxy_kwargs)
                # Patch with instructor
                return instructor.patch(raw_client)
            except Exception as e:
                print(f"Warning: Could not use DeepSeek client: {e}. Trying from_provider instead.")
                # Fall back to from_provider
                try:
                    return instructor.from_provider(f"{provider}/{model_name}")
                except Exception as inner_e:
                    error_msg = f"Failed to initialize DeepSeek client: {e}, {inner_e}"
                    raise ValueError(error_msg)
        elif provider == "gemini":
            # Gemini code goes here
            raise ValueError("Gemini Provider will be added in future releases")
        else:
            # For other providers, use from_provider with empty kwargs
            try:
                # Create client using from_provider
                return instructor.from_provider(f"{provider}/{model_name}")
            except Exception as e:
                raise ValueError(f"Unsupported provider {provider}: {e}")

    def get_model_kwargs(self) -> Dict[str, Any]:
        """
        Get model-specific kwargs from the model configuration,
        but exclude api_key which should be set in the client.

        Returns:
            Dictionary of kwargs for the specific model provider
        """
        model_kwargs = self.model_config.get_model_kwargs()

        # Remove api_key if present to avoid passing it in create() calls
        if "api_key" in model_kwargs:
            del model_kwargs["api_key"]

        return model_kwargs

def create_llm_client(
    model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,  # Add DeepSeek API key
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client with the specified configuration.

    Args:
        model_config: Configuration for the LLM model to use
        openai_api_key: Optional API key for OpenAI
        anthropic_api_key: Optional API key for Anthropic
        deepseek_api_key: Optional API key for DeepSeek
        **kwargs: Additional keyword arguments to pass to the client

    Returns:
        Initialized LLMClient instance
    """
    return LLMClient(
        model_config=model_config,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,  # Pass DeepSeek API key
        **kwargs
    )
