"""
Schema definitions for the syda library.
Contains Pydantic models used for data validation and configuration.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, Literal


class ProxyConfig(BaseModel):
    """
    Configuration for API proxy settings, commonly used in enterprise environments.
    """
    
    # Base URL for the proxy service
    base_url: Optional[str] = Field(None, description="Base URL for the proxy service (e.g. 'https://ai-proxy.company.com/v1')")
    
    # Additional headers to include in requests
    headers: Optional[Dict[str, str]] = Field(None, description="Additional HTTP headers to include in requests to the proxy")
    
    # Additional query parameters to include in URL
    params: Optional[Dict[str, Any]] = Field(None, description="Additional query parameters to include in the URL")
    
    # Path modification (some proxies require a different path structure)
    path_format: Optional[str] = Field(None, description="Optional format string for proxy path, e.g. '/proxy/{provider}/{endpoint}'")
    
    def get_proxy_kwargs(self) -> Dict[str, Any]:
        """Get proxy-specific kwargs for API client initialization."""
        kwargs = {}
        
        # Handle base_url and query parameters
        if self.base_url:
            base_url = self.base_url
            
            # Add query parameters to the base URL if provided
            if self.params:
                # Convert all param values to strings
                params = {k: str(v) for k, v in self.params.items()}
                
                # Create the query string
                from urllib.parse import urlencode
                query_string = urlencode(params)
                
                # Append to base_url with ? or & depending on whether URL already has params
                if "?" in base_url:
                    base_url += "&" + query_string
                else:
                    base_url += "?" + query_string
                    
            kwargs["base_url"] = base_url
            
        # Handle custom headers
        if self.headers:
            kwargs["default_headers"] = self.headers
            
        return kwargs


class ModelConfig(BaseModel):
    """
    Configuration for AI model settings used by the SyntheticDataGenerator.
    
    This class provides a structured way to define the model and its parameters,
    supporting both OpenAI and Anthropic models with provider-specific settings.
    """
    
    # Model provider and name
    provider: Literal["openai", "anthropic"] = "openai"
    model_name: str = "gpt-4"
    
    # Common parameters
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Controls randomness: 0.0 is deterministic, higher values are more random")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    
    # OpenAI specific parameters
    seed: Optional[int] = Field(None, description="Random seed for reproducibility (OpenAI only)")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Format for responses (OpenAI only)")
    
    # Anthropic specific parameters
    top_k: Optional[int] = Field(None, description="Top K sampling parameter (Anthropic only)")
    max_tokens_to_sample: Optional[int] = Field(None, description="Maximum tokens to generate (Anthropic only)")
    
    # Proxy configuration
    proxy: Optional[ProxyConfig] = Field(None, description="Optional proxy configuration for API requests")
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model-specific kwargs for API calls."""
        # Start with common parameters
        kwargs = {"temperature": self.temperature}
        
        # Always include the model name, which is required for OpenAI and used for other providers
        kwargs["model"] = self.model_name
        
        # Add provider-specific parameters
        if self.provider == "openai":
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p:
                kwargs["top_p"] = self.top_p
            if self.seed:
                kwargs["seed"] = self.seed
            if self.response_format:
                kwargs["response_format"] = self.response_format
        
        elif self.provider == "anthropic":
            # The updated Anthropic API via instructor now uses the same parameter names as OpenAI
            # for better compatibility across providers
            if self.max_tokens_to_sample or self.max_tokens:
                # Use max_tokens for consistency with OpenAI naming
                kwargs["max_tokens"] = self.max_tokens_to_sample or self.max_tokens
                # Remove the legacy parameter to avoid errors
                if "max_tokens_to_sample" in kwargs:
                    del kwargs["max_tokens_to_sample"]
            if self.top_p:
                kwargs["top_p"] = self.top_p
            if self.top_k:
                # This might not be supported in the current instructor integration
                # but we'll keep it for future compatibility
                kwargs["top_k"] = self.top_k
            
            # For Anthropic via instructor, we need to use 'model' for consistency
            kwargs["model"] = self.model_name
        
        return kwargs
