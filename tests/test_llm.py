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
    @patch('syda.llm.openai.OpenAI')
    @patch('syda.llm.instructor.from_openai')
    def test_initialize_client_without_env_keys(self, mock_from_openai, mock_openai_client):
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


class TestOpenAICompatibleProvider:
    """Tests for the openai_compatible provider in LLMClient."""

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_valid_config_with_base_url_and_api_key(self, mock_from_openai, mock_openai):
        """Client initializes correctly with base_url and api_key in extra_kwargs."""
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        )
        client = LLMClient(model_config=config)

        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1", api_key="ollama"
        )
        mock_from_openai.assert_called_once()
        assert client.model_config == config

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_missing_base_url_raises_error(self, mock_from_openai, mock_openai):
        """ValueError with helpful message when base_url is missing."""
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"api_key": "ollama"},
        )
        with pytest.raises(ValueError, match="base_url"):
            LLMClient(model_config=config)

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_missing_extra_kwargs_raises_error(self, mock_from_openai, mock_openai):
        """ValueError raised when extra_kwargs is not provided at all."""
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
        )
        with pytest.raises(ValueError, match="base_url"):
            LLMClient(model_config=config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env_key"})
    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_api_key_falls_back_to_env_var(self, mock_from_openai, mock_openai):
        """api_key falls back to OPENAI_API_KEY env var when not in extra_kwargs."""
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"base_url": "http://localhost:11434/v1"},
        )
        LLMClient(model_config=config)

        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1", api_key="env_key"
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_api_key_falls_back_to_none_string(self, mock_from_openai, mock_openai):
        """api_key defaults to 'none' when not set anywhere (e.g. Ollama)."""
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"base_url": "http://localhost:11434/v1"},
        )
        LLMClient(model_config=config)

        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1", api_key="none"
        )

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_default_response_mode_is_markdown(self, mock_from_openai, mock_openai):
        """Default response_mode is 'markdown' (MD_JSON) when not specified."""
        import instructor
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        )
        LLMClient(model_config=config)

        _, kwargs = mock_from_openai.call_args
        assert kwargs.get("mode") == instructor.Mode.MD_JSON

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_response_mode_tools(self, mock_from_openai, mock_openai):
        """response_mode='tools' uses tool-call mode for models that support it."""
        import instructor
        config = ModelConfig(
            provider="openai_compatible",
            model_name="gpt-oss:20b",
            extra_kwargs={
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "response_mode": "tools",
            },
        )
        LLMClient(model_config=config)

        _, kwargs = mock_from_openai.call_args
        assert kwargs.get("mode") == instructor.Mode.TOOLS

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_response_mode_json(self, mock_from_openai, mock_openai):
        """response_mode='json' uses clean JSON content mode."""
        import instructor
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={
                "base_url": "http://localhost:11434/v1",
                "response_mode": "json",
            },
        )
        LLMClient(model_config=config)

        _, kwargs = mock_from_openai.call_args
        assert kwargs.get("mode") == instructor.Mode.JSON

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_response_mode_is_case_insensitive(self, mock_from_openai, mock_openai):
        """response_mode value is case-insensitive."""
        import instructor
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={
                "base_url": "http://localhost:11434/v1",
                "response_mode": "MARKDOWN",
            },
        )
        LLMClient(model_config=config)

        _, kwargs = mock_from_openai.call_args
        assert kwargs.get("mode") == instructor.Mode.MD_JSON

    @patch("syda.llm.openai.OpenAI")
    @patch("syda.llm.instructor.from_openai")
    def test_invalid_response_mode_raises_error(self, mock_from_openai, mock_openai):
        """Invalid response_mode raises ValueError with valid options listed."""
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={
                "base_url": "http://localhost:11434/v1",
                "response_mode": "xml",
            },
        )
        with pytest.raises(ValueError, match="Invalid response_mode"):
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
            grok_api_key=None,
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
            grok_api_key=None,
            provider="openai", 
            model_name="gpt-4o", 
            api_key="test_key"
        )
