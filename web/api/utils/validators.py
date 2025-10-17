"""
Common validators
"""
from typing import Dict, Any


def validate_extra_kwargs(extra_kwargs: Dict[str, str]) -> Dict[str, str]:
    """
    Validate extra_kwargs dictionary
    
    Args:
        extra_kwargs: Dictionary of extra keyword arguments
        
    Returns:
        Validated dictionary
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(extra_kwargs, dict):
        raise ValueError("extra_kwargs must be a dictionary")
    
    # Ensure all values are strings
    for key, value in extra_kwargs.items():
        if not isinstance(key, str):
            raise ValueError(f"All keys must be strings, got {type(key)}")
        if not isinstance(value, str):
            raise ValueError(f"All values must be strings, got {type(value)} for key '{key}'")
    
    return extra_kwargs


def validate_provider_key(key: str) -> str:
    """
    Validate provider key
    
    Args:
        key: Provider key (anthropic, openai, etc.)
        
    Returns:
        Validated key
        
    Raises:
        ValueError: If validation fails
    """
    valid_keys = ["anthropic", "openai", "gemini", "azureopenai", "grok"]
    
    if key not in valid_keys:
        raise ValueError(f"Invalid provider key. Must be one of: {', '.join(valid_keys)}")
    
    return key

