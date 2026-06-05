"""
Tests for the llm module (pydantic-ai backend).
"""
import pytest
from unittest.mock import patch, MagicMock

from syda.llm import LLMClient, create_llm_client
from syda.schemas import ModelConfig


def _mock_model():
    return MagicMock(name="PydanticAIModel")


class TestLLMClientInit:
    """LLMClient initialises correctly for every supported provider."""

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_default_config(self, mock_build):
        client = LLMClient()
        assert client.model_config.provider == "anthropic"
        mock_build.assert_called_once()

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_openai_provider(self, mock_build):
        config = ModelConfig(provider="openai", model_name="gpt-4o")
        client = LLMClient(model_config=config)
        assert client.model_config == config
        mock_build.assert_called_once_with(config)

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_anthropic_provider(self, mock_build):
        config = ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001")
        client = LLMClient(model_config=config)
        assert client.model_config == config

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_gemini_provider(self, mock_build):
        config = ModelConfig(provider="gemini", model_name="gemini-2.0-flash")
        client = LLMClient(model_config=config)
        assert client.model_config == config

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_dict_model_config(self, mock_build):
        client = LLMClient(model_config={"provider": "openai", "model_name": "gpt-4o-mini"})
        assert client.model_config.provider == "openai"
        assert client.model_config.model_name == "gpt-4o-mini"

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_api_keys_set_env_vars(self, mock_build):
        import os
        LLMClient(
            model_config=ModelConfig(provider="openai", model_name="gpt-4o"),
            openai_api_key="test-openai",
            anthropic_api_key="test-anthropic",
        )
        assert os.environ.get("OPENAI_API_KEY") == "test-openai"
        assert os.environ.get("ANTHROPIC_API_KEY") == "test-anthropic"

    def test_no_client_attribute(self):
        """LLMClient must NOT expose a .client attribute (instructor is gone)."""
        with patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model()):
            client = LLMClient()
        assert not hasattr(client, "client")


class TestCreateAgent:
    """LLMClient.create_agent returns a usable pydantic-ai Agent."""

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_returns_agent(self, mock_build):
        from pydantic_ai import Agent

        client = LLMClient()
        agent = client.create_agent(output_type=str, system_prompt="test")
        assert isinstance(agent, Agent)

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_create_agent_with_system_prompt(self, mock_build):
        from pydantic_ai import Agent

        client = LLMClient()
        agent = client.create_agent(output_type=list, system_prompt="You are helpful.")
        assert isinstance(agent, Agent)


class TestGetModelSettings:
    """LLMClient.get_model_settings extracts ModelConfig parameters correctly."""

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_returns_none_when_no_settings(self, _):
        client = LLMClient(model_config=ModelConfig())
        assert client.get_model_settings() is None

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_temperature_included(self, _):
        config = ModelConfig(provider="openai", model_name="gpt-4o", temperature=0.7)
        client = LLMClient(model_config=config)
        settings = client.get_model_settings()
        assert settings is not None
        assert settings.get("temperature") == 0.7

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_max_tokens_included(self, _):
        config = ModelConfig(provider="openai", model_name="gpt-4o", max_tokens=1024)
        client = LLMClient(model_config=config)
        settings = client.get_model_settings()
        assert settings is not None
        assert settings.get("max_tokens") == 1024

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_max_tokens_to_sample_as_max_tokens(self, _):
        config = ModelConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
            max_tokens_to_sample=2048,
        )
        client = LLMClient(model_config=config)
        settings = client.get_model_settings()
        assert settings is not None
        assert settings.get("max_tokens") == 2048


class TestOpenAICompatibleProvider:
    """openai_compatible provider requires base_url in extra_kwargs."""

    def test_valid_config_with_base_url(self):
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        )
        with patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model()):
            client = LLMClient(model_config=config)
        assert client.model_config == config

    def test_missing_base_url_raises_error(self):
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"api_key": "ollama"},
        )
        with pytest.raises(ValueError, match="base_url"):
            LLMClient(model_config=config)

    def test_missing_extra_kwargs_raises_error(self):
        config = ModelConfig(provider="openai_compatible", model_name="llama3")
        with pytest.raises(ValueError, match="base_url"):
            LLMClient(model_config=config)

    def test_api_key_defaults_to_none_string(self):
        """When api_key omitted, 'none' is passed so pydantic-ai doesn't error."""
        config = ModelConfig(
            provider="openai_compatible",
            model_name="llama3",
            extra_kwargs={"base_url": "http://localhost:11434/v1"},
        )
        # We can verify by checking the model is built without raising
        with patch(
            "pydantic_ai.models.openai.OpenAIChatModel", autospec=True
        ) as mock_chat_model, patch(
            "pydantic_ai.providers.openai.OpenAIProvider", autospec=True
        ) as mock_provider:
            LLMClient(model_config=config)
            _, provider_kwargs = mock_provider.call_args
            assert provider_kwargs.get("api_key") == "none"


class TestCreateLLMClient:
    """create_llm_client factory delegates to LLMClient."""

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_returns_llm_client(self, _):
        config = ModelConfig(provider="openai", model_name="gpt-4o")
        client = create_llm_client(model_config=config)
        assert isinstance(client, LLMClient)

    @patch("syda.llm._build_pydantic_ai_model", return_value=_mock_model())
    def test_passes_through_api_keys(self, _):
        import os
        config = ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001")
        create_llm_client(model_config=config, anthropic_api_key="sk-test")
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-test"
