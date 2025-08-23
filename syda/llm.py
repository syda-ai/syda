"""
Unified LLM client initialization for different providers.
Provides a standard interface for creating LLM clients with instructor integration.
"""

import os
import instructor
import openai
from google import genai
from typing import Optional, Dict, Any, Union, List
from .schemas import ModelConfig, ProxyConfig

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
        gemini_api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize an LLM client for the specified model provider.
        
        Args:
            model_config: Configuration for the LLM model to use
            openai_api_key: Optional API key for OpenAI
            anthropic_api_key: Optional API key for Anthropic
            gemini_api_key: Optional API key for Gemini
            **kwargs: Additional keyword arguments to pass to the client
        """
        # Set up API keys from arguments or environment variables
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
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
                
        elif provider == "gemini":
            # Set up environment variable for Gemini instead of passing directly
            if self.gemini_api_key:
                os.environ["GEMINI_API_KEY"] = self.gemini_api_key

            # Get proxy configuration
            proxy_kwargs = {}
            if self.model_config.proxy:
                proxy_kwargs = self.model_config.proxy.get_proxy_kwargs()

            # Create raw client
            raw_client = genai.Client(**proxy_kwargs)

            # Patch with instructor
            return instructor.from_genai(raw_client)

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
    gemini_api_key: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client with the specified configuration.
    
    Args:
        model_config: Configuration for the LLM model to use
        openai_api_key: Optional API key for OpenAI
        anthropic_api_key: Optional API key for Anthropic
        gemini_api_key: Optional API key for Gemini
        **kwargs: Additional keyword arguments to pass to the client
    
    Returns:
        Initialized LLMClient instance
    """
    return LLMClient(
        model_config=model_config,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        **kwargs
    )
