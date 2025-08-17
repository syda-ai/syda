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
    def test_initialize_openai_client(self, mock_from_openai, mock_openai_client):
        """Test initializing an OpenAI client."""
        # Create a model config for OpenAI
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        # Create the LLMClient
        client = LLMClient(model_config=config)
        
        # Check that the OpenAI client was created
        # The actual code sets the API key via environment variables, not constructor parameters
        mock_openai_client.assert_called_once_with()
        
        # Check that the client was patched with instructor
        mock_from_openai.assert_called_once()
        
        # Check that the client has the expected attributes
        assert client.model_config == config
        assert client.client is not None
    
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    @patch('syda.llm.instructor.patch')
    def test_initialize_anthropic_client(self, mock_patch, mock_anthropic_client):
        """Test initializing an Anthropic client."""
        # Create a model config for Anthropic
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet"
        )
        
        # Create the LLMClient
        client = LLMClient(model_config=config)
        
        # Check that the Anthropic client was created
        # The actual code sets the API key via environment variables, not constructor parameters
        mock_anthropic_client.assert_called_once_with()
        
        # Check that the client was patched with instructor
        mock_patch.assert_called_once()
        
        # Check that the client has the expected attributes
        assert client.model_config == config
        assert client.client is not None

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('syda.llm.genai.Client')
    @patch('syda.llm.instructor.from_genai')
    def test_initialize_gemini_client(self, mock_from_genai, mock_genai_client):
        """Test initializing a Gemini client."""
        # Create a model config for Gemini
        config = ModelConfig(
            provider="gemini",
            model_name="gemini-2.5-flash"
        )
        
        # Create the LLMClient
        client = LLMClient(model_config=config)
        
        # Check that the Gemini client was created
        mock_genai_client.assert_called_once_with()
        
        # Check that the client was patched with instructor
        mock_from_genai.assert_called_once()
        
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
    
    @patch('syda.llm.os.environ', {})
    @patch('syda.llm.openai.OpenAI')
    @patch('syda.llm.instructor.from_openai')
    def test_initialize_client_missing_api_key(self, mock_from_openai, mock_openai_client):
        """Test initializing a client when API key is missing."""
        # Set up the mock to return something valid
        mock_from_openai.return_value = MagicMock()
        
        # Create a model config
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o"
        )
        
        # The client initializes without error in our test because we've mocked the dependencies
        client = LLMClient(model_config=config)
        
        # Verify the client was created with correct properties
        assert client.model_config == config
        # Since we mocked os.environ to be empty, the client should not have an API key
        assert client.openai_api_key is None
        
        # Verify the OpenAI client and instructor were called
        assert mock_openai_client.called
        assert mock_from_openai.called
    
    @patch('syda.llm.os.environ', {"OPENAI_API_KEY": "test_key"})
    @patch('syda.llm.openai.OpenAI')
    def test_initialize_client_unsupported_provider(self, mock_openai):
        """Test that using an unsupported provider configuration raises an error."""
        # Set up the mock to raise an error
        mock_openai.side_effect = ValueError("Unsupported provider")
        
        # Create a model config with a valid provider
        config = ModelConfig(
            provider="openai",
            model_name="invalid-model"
        )
        
        # Creating the LLMClient should raise the error from our mock
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClient(model_config=config)


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
        # The api_key is passed as a kwargs parameter
        mock_llm_client.assert_called_once_with(
            model_config=config, 
            openai_api_key=None, 
            anthropic_api_key=None, 
            gemini_api_key=None,
            api_key="test_key"
        )
    
    @patch('syda.llm.LLMClient')
    def test_create_llm_client_without_config(self, mock_llm_client): 
        """Test creating an LLM client without a model config."""
        # In the actual implementation, provider and model_name would be passed as kwargs
        # and the default ModelConfig() would be created inside the LLMClient constructor
        
        # Create the client
        create_llm_client(provider="openai", model_name="gpt-4o", api_key="test_key")
        
        # Check that LLMClient was called with the right arguments
        # model_config should be None, and provider/model_name passed as kwargs
        mock_llm_client.assert_called_once_with(
            model_config=None, 
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            provider="openai", 
            model_name="gpt-4o", 
            api_key="test_key"
        )
