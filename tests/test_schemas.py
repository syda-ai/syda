"""
Tests for the schemas module.
"""
import pytest
from unittest.mock import patch

from syda.schemas import ModelConfig


class TestModelConfig:
    """Tests for the ModelConfig class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig(provider="openai", model_name="gpt-4")
        
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.2
        assert config.max_completion_tokens == 2048
        assert config.top_p == 1.0
        assert config.stream is False
    
    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            temperature=0.5,
            max_completion_tokens=4096,
            top_p=0.9,
            stream=True
        )
        
        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-5-sonnet"
        assert config.temperature == 0.5
        assert config.max_completion_tokens == 4096
        assert config.top_p == 0.9
        assert config.stream is True
    
    def test_get_model_kwargs_for_openai(self):
        """Test getting model kwargs for OpenAI."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.7,
            max_completion_tokens=1000,
            top_p=0.8,
            stream=True
        )
        
        kwargs = config.get_model_kwargs()
        
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert kwargs["top_p"] == 0.8
        assert kwargs["stream"] is True
    
    def test_get_model_kwargs_for_anthropic(self):
        """Test getting model kwargs for Anthropic."""
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            temperature=0.7,
            max_completion_tokens=1000,
            top_p=0.8,
            stream=True
        )
        
        kwargs = config.get_model_kwargs()
        
        assert kwargs["model"] == "claude-3-5-sonnet"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert kwargs["top_p"] == 0.8
        assert kwargs["stream"] is True
    
    def test_get_model_kwargs_with_unsupported_provider(self):
        """Test getting model kwargs with unsupported provider."""
        config = ModelConfig(
            provider="unsupported",
            model_name="test-model"
        )
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            config.get_model_kwargs()
    
    @patch("syda.schemas.ModelConfig.get_model_kwargs")
    def test_get_model_kwargs_is_called(self, mock_get_model_kwargs):
        """Test that get_model_kwargs is called when accessing model_kwargs."""
        mock_get_model_kwargs.return_value = {"test_key": "test_value"}
        
        config = ModelConfig(provider="openai", model_name="gpt-4")
        kwargs = config.model_kwargs
        
        mock_get_model_kwargs.assert_called_once()
        assert kwargs == {"test_key": "test_value"}
