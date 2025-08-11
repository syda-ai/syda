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
        config = ModelConfig()
        
        # Test default values
        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-5-haiku-20241022"
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.stream is False
        assert config.seed is None
        assert config.response_format is None
        assert config.max_completion_tokens is None
        assert config.top_k is None
        assert config.max_tokens_to_sample is None
        assert config.proxy is None
    
    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            temperature=0.5,
            max_tokens=4096,
            stream=True,
            top_k=40,
            max_tokens_to_sample=2000
        )
        
        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-5-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096
        assert config.stream is True
        assert config.top_k == 40
        assert config.max_tokens_to_sample == 2000
    
    def test_openai_specific_values(self):
        """Test OpenAI-specific parameters."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            seed=42,
            response_format={"type": "json_object"},
            max_completion_tokens=800
        )
        
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.seed == 42
        assert config.response_format == {"type": "json_object"}
        assert config.max_completion_tokens == 800
    
    def test_get_model_kwargs_for_openai(self):
        """Test getting model kwargs for OpenAI."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            seed=42,
            response_format={"type": "json_object"},
            max_completion_tokens=800
        )
        
        kwargs = config.get_model_kwargs()
        
        assert kwargs["model"] == "gpt-4"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert kwargs["seed"] == 42
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["max_completion_tokens"] == 800
    
    def test_get_model_kwargs_for_anthropic(self):
        """Test getting model kwargs for Anthropic."""
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            temperature=0.7,
            max_tokens=1000,
            top_k=40,
            max_tokens_to_sample=2000
        )
        
        kwargs = config.get_model_kwargs()
        
        assert kwargs["model"] == "claude-3-5-sonnet"
        assert kwargs["temperature"] == 0.7
        # max_tokens_to_sample should override max_tokens for Anthropic
        assert kwargs["max_tokens"] == 2000
        assert kwargs["top_k"] == 40
    
    def test_get_model_kwargs_anthropic_without_max_tokens_to_sample(self):
        """Test Anthropic kwargs when max_tokens_to_sample is not provided."""
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-haiku-20241022",
            temperature=0.5,
            max_tokens=1500
        )
        
        kwargs = config.get_model_kwargs()
        
        assert kwargs["model"] == "claude-3-5-haiku-20241022"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 1500
    
    def test_get_model_kwargs_with_default_values(self):
        """Test that default None values are handled correctly in kwargs."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-3.5-turbo"
            # temperature and max_tokens will be None (default)
        )
        
        kwargs = config.get_model_kwargs()
        
        assert kwargs["model"] == "gpt-3.5-turbo"
        # None values should not be included in kwargs
        assert "temperature" not in kwargs or kwargs.get("temperature") is None
        assert "max_tokens" not in kwargs or kwargs.get("max_tokens") is None
        assert "seed" not in kwargs  # seed is None and should not be included
    
    def test_get_model_kwargs_with_unsupported_provider(self):
        """Test validation for unsupported provider."""
        with pytest.raises(ValueError) as excinfo:
            ModelConfig(
                provider="unsupported",  # This should fail validation
                model_name="test-model"
            )
        # Verify that the error mentions provider validation
        assert "provider" in str(excinfo.value).lower()
    
    def test_temperature_validation(self):
        """Test temperature validation (should be between 0.0 and 1.0)."""
        # Valid temperature
        config = ModelConfig(temperature=0.7)
        assert config.temperature == 0.7
        
        # Invalid temperature - too high
        with pytest.raises(ValueError):
            ModelConfig(temperature=1.5)
        
        # Invalid temperature - negative
        with pytest.raises(ValueError):
            ModelConfig(temperature=-0.1)
        
        # Edge cases - valid
        config_zero = ModelConfig(temperature=0.0)
        assert config_zero.temperature == 0.0
        
        config_one = ModelConfig(temperature=1.0)
        assert config_one.temperature == 1.0
    
    @patch("syda.schemas.ModelConfig.get_model_kwargs")
    def test_get_model_kwargs_is_called(self, mock_get_model_kwargs):
        """Test that get_model_kwargs method works as expected."""
        mock_get_model_kwargs.return_value = {"test_key": "test_value"}
        
        config = ModelConfig(provider="openai", model_name="gpt-4")
        kwargs = config.get_model_kwargs()
        
        mock_get_model_kwargs.assert_called_once()
        assert kwargs == {"test_key": "test_value"}
