"""
Tests for the llm module.
"""
import pytest
from unittest.mock import patch, MagicMock

from syda.llm import LLMClient, create_llm_client
from syda.schemas import ModelConfig


class TestLLMClient:
    """Tests for the LLMClient class."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    @patch('syda.llm.openai.OpenAI')
    @patch('syda.llm.instructor.from_openai')
    def test_initialize_openai_client(self, mock_from_openai, mock_openai_client, mock_env):
        """Test initializing an OpenAI client."""
        # Create a model config for OpenAI
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        # Create the LLMClient
        client = LLMClient(model_config=config)
        
        # Check that the OpenAI client was created with the API key
        mock_openai_client.assert_called_once_with(api_key='test_key')
        
        # Check that the client was patched with instructor
        mock_from_openai.assert_called_once()
        
        # Check that the client has the expected attributes
        assert client.model_config == config
        assert client.client is not None
    
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('syda.llm.anthropic.Anthropic')
    @patch('syda.llm.instructor.patch')
    def test_initialize_anthropic_client(self, mock_patch, mock_anthropic_client, mock_env):
        """Test initializing an Anthropic client."""
        # Create a model config for Anthropic
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet"
        )
        
        # Create the LLMClient
        client = LLMClient(model_config=config)
        
        # Check that the Anthropic client was created with the API key
        mock_anthropic_client.assert_called_once_with(api_key='test_key')
        
        # Check that the client was patched with instructor
        mock_patch.assert_called_once()
        
        # Check that the client has the expected attributes
        assert client.model_config == config
        assert client.client is not None
    
    @patch.dict('os.environ', {})
    def test_initialize_client_without_env_keys(self):
        """Test initializing a client without environment variables."""
        # Create a model config
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        # Create the LLMClient with an explicit API key
        client = LLMClient(model_config=config, api_key="explicit_key")
        
        # Check that the client has the expected attributes
        assert client.model_config == config
        assert client.client is not None
    
    @patch.dict('os.environ', {})
    def test_initialize_client_missing_api_key(self):
        """Test that initializing a client without an API key raises an error."""
        # Create a model config
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        # Creating the LLMClient without an API key should raise an error
        with pytest.raises(ValueError, match="API key not provided"):
            LLMClient(model_config=config)
    
    def test_initialize_client_unsupported_provider(self):
        """Test that initializing a client with an unsupported provider raises an error."""
        # Create a model config with an unsupported provider
        config = ModelConfig(
            provider="unsupported",
            model_name="model"
        )
        
        # Creating the LLMClient with an unsupported provider should raise an error
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClient(model_config=config, api_key="test_key")


class TestCreateLLMClient:
    """Tests for the create_llm_client function."""
    
    @patch('syda.llm.LLMClient')
    def test_create_llm_client_with_config(self, mock_llm_client):
        """Test creating an LLM client with a model config."""
        # Create a model config
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        # Create the client
        create_llm_client(model_config=config, api_key="test_key")
        
        # Check that LLMClient was called with the right arguments
        mock_llm_client.assert_called_once_with(model_config=config, api_key="test_key")
    
    @patch('syda.llm.ModelConfig')
    @patch('syda.llm.LLMClient')
    def test_create_llm_client_without_config(self, mock_llm_client, mock_model_config):
        """Test creating an LLM client without a model config."""
        # Mock ModelConfig to return a test config
        test_config = ModelConfig(provider="openai", model_name="gpt-4o")
        mock_model_config.return_value = test_config
        
        # Create the client
        create_llm_client(provider="openai", model_name="gpt-4o", api_key="test_key")
        
        # Check that ModelConfig was called with the right arguments
        mock_model_config.assert_called_once_with(provider="openai", model_name="gpt-4o")
        
        # Check that LLMClient was called with the right arguments
        mock_llm_client.assert_called_once_with(model_config=test_config, api_key="test_key")
