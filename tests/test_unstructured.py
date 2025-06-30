"""
Tests for the unstructured module.
"""
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open, call

from syda.unstructured import (
    UnstructuredDataGenerator,
    save_text,
    save_texts
)
from syda.schemas import ModelConfig


@pytest.fixture
def model_config():
    """Return a sample model config."""
    return ModelConfig(provider="openai", model_name="gpt-4")


class TestUnstructuredDataGenerator:
    """Tests for the UnstructuredDataGenerator class."""
    
    def test_initialization(self, model_config):
        """Test initialization of UnstructuredDataGenerator."""
        generator = UnstructuredDataGenerator(model_config=model_config, api_key="test_key")
        
        # Check that attributes are properly initialized
        assert generator.model_config == model_config
        assert generator.api_key == "test_key"
        assert generator.client is not None
    
    @patch("syda.unstructured.create_llm_client")
    def test_single_text_generation(self, mock_create_client, model_config):
        """Test generating a single text."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.complete.return_value = "Generated text content"
        
        # Create the generator
        generator = UnstructuredDataGenerator(model_config=model_config, api_key="test_key")
        
        # Generate text
        text = generator.generate_text("Generate a poem about AI", 
                                      system_prompt="You are a helpful assistant")
        
        # Check that the client was called
        mock_create_client.assert_called_once()
        mock_client.complete.assert_called_once()
        
        # Check that the text was returned
        assert text == "Generated text content"
    
    @patch("syda.unstructured.create_llm_client")
    def test_multiple_text_generation(self, mock_create_client, model_config):
        """Test generating multiple texts."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.complete.side_effect = ["Text 1", "Text 2"]
        
        # Create the generator
        generator = UnstructuredDataGenerator(model_config=model_config, api_key="test_key")
        
        # Generate texts
        texts = generator.generate_texts(
            "Generate a story",
            count=2,
            system_prompt="You are a creative writer"
        )
        
        # Check that the client was called multiple times
        assert mock_client.complete.call_count == 2
        
        # Check that texts were returned
        assert texts == ["Text 1", "Text 2"]
    
    @patch("syda.unstructured.create_llm_client")
    def test_client_error_handling(self, mock_create_client, model_config):
        """Test error handling when the client fails."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.complete.side_effect = Exception("API Error")
        
        # Create the generator
        generator = UnstructuredDataGenerator(model_config=model_config, api_key="test_key")
        
        # Generate text with error handling
        with pytest.raises(Exception, match="API Error"):
            generator.generate_text("Generate a poem")
    
    @patch("syda.unstructured.save_text")
    @patch("syda.unstructured.create_llm_client")
    def test_generate_and_save_text(self, mock_create_client, mock_save_text, model_config):
        """Test generating and saving text."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.complete.return_value = "Generated text"
        
        # Create the generator
        generator = UnstructuredDataGenerator(model_config=model_config, api_key="test_key")
        
        # Generate and save text
        filename = generator.generate_and_save(
            "Generate a poem",
            output_path="output/poem.txt",
            system_prompt="You are a poet"
        )
        
        # Check that save_text was called
        mock_save_text.assert_called_once_with("Generated text", "output/poem.txt")
        
        # Check that the filename was returned
        assert filename == "output/poem.txt"
    
    @patch("syda.unstructured.save_texts")
    @patch("syda.unstructured.create_llm_client")
    def test_generate_and_save_multiple_texts(self, mock_create_client, mock_save_texts, model_config):
        """Test generating and saving multiple texts."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.complete.side_effect = ["Text 1", "Text 2"]
        
        # Create the generator
        generator = UnstructuredDataGenerator(model_config=model_config, api_key="test_key")
        
        # Generate and save texts
        filenames = generator.generate_and_save_multiple(
            "Generate stories",
            count=2,
            output_dir="output",
            prefix="story",
            extension=".txt",
            system_prompt="You are a writer"
        )
        
        # Check that save_texts was called
        mock_save_texts.assert_called_once_with(
            ["Text 1", "Text 2"], "output", "story", ".txt"
        )
        
        # Check that filenames were returned
        assert isinstance(filenames, list)
    
    @patch("syda.unstructured.create_llm_client")
    def test_system_prompt_default(self, mock_create_client, model_config):
        """Test default system prompt."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        # Create the generator
        generator = UnstructuredDataGenerator(model_config=model_config, api_key="test_key")
        
        # Generate text without specifying a system prompt
        generator.generate_text("Generate a poem")
        
        # Check that the client was called with the default system prompt
        mock_client.complete.assert_called_once()
        args, kwargs = mock_client.complete.call_args
        system_prompt = kwargs.get('system')
        assert system_prompt is not None
        assert "helpful assistant" in system_prompt.lower()


class TestSaveText:
    """Tests for the save_text function."""
    
    @patch("os.path.dirname")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_text(self, mock_file, mock_makedirs, mock_dirname):
        """Test saving text to a file."""
        # Configure mocks
        mock_dirname.return_value = "output"
        
        # Save text
        result = save_text("Text content", "output/text.txt")
        
        # Check that the directory was created
        mock_makedirs.assert_called_once_with("output", exist_ok=True)
        
        # Check that the file was written
        mock_file.assert_called_once_with("output/text.txt", "w")
        mock_file().write.assert_called_once_with("Text content")
        
        # Check that the path was returned
        assert result == "output/text.txt"
    
    @patch("os.path.dirname")
    @patch("os.makedirs")
    @patch("builtins.open", side_effect=IOError("File error"))
    def test_save_text_error(self, mock_file, mock_makedirs, mock_dirname):
        """Test error handling when saving text fails."""
        # Configure mocks
        mock_dirname.return_value = "output"
        
        # Try to save text
        with pytest.raises(IOError, match="File error"):
            save_text("Text content", "output/text.txt")


class TestSaveTexts:
    """Tests for the save_texts function."""
    
    @patch("os.makedirs")
    @patch("syda.unstructured.save_text")
    def test_save_texts(self, mock_save_text, mock_makedirs):
        """Test saving multiple texts."""
        # Configure the save_text mock
        mock_save_text.side_effect = lambda text, path: path
        
        # Save texts
        paths = save_texts(
            ["Text 1", "Text 2", "Text 3"],
            "output",
            "file",
            ".txt"
        )
        
        # Check that the directory was created
        mock_makedirs.assert_called_once_with("output", exist_ok=True)
        
        # Check that save_text was called for each text
        assert mock_save_text.call_count == 3
        mock_save_text.assert_has_calls([
            call("Text 1", os.path.join("output", "file_0.txt")),
            call("Text 2", os.path.join("output", "file_1.txt")),
            call("Text 3", os.path.join("output", "file_2.txt"))
        ])
        
        # Check that paths were returned
        assert len(paths) == 3
        assert all(path.startswith("output/file_") and path.endswith(".txt") for path in paths)
    
    @patch("os.makedirs")
    @patch("syda.unstructured.save_text")
    def test_save_texts_empty_list(self, mock_save_text, mock_makedirs):
        """Test saving an empty list of texts."""
        # Save an empty list of texts
        paths = save_texts([], "output", "file", ".txt")
        
        # Check that the directory was created
        mock_makedirs.assert_called_once_with("output", exist_ok=True)
        
        # Check that save_text was not called
        mock_save_text.assert_not_called()
        
        # Check that an empty list was returned
        assert paths == []
    
    @patch("os.makedirs", side_effect=OSError("Directory error"))
    def test_save_texts_directory_error(self, mock_makedirs):
        """Test error handling when creating the directory fails."""
        # Try to save texts
        with pytest.raises(OSError, match="Directory error"):
            save_texts(["Text"], "output", "file", ".txt")
    
    @patch("os.makedirs")
    @patch("syda.unstructured.save_text")
    def test_save_texts_custom_extension(self, mock_save_text, mock_makedirs):
        """Test saving texts with a custom extension."""
        # Configure the save_text mock
        mock_save_text.side_effect = lambda text, path: path
        
        # Save texts with a custom extension
        paths = save_texts(
            ["Text 1", "Text 2"],
            "output",
            "document",
            ".md"
        )
        
        # Check that save_text was called with the custom extension
        mock_save_text.assert_has_calls([
            call("Text 1", os.path.join("output", "document_0.md")),
            call("Text 2", os.path.join("output", "document_1.md"))
        ])
        
        # Check that paths have the custom extension
        assert all(path.endswith(".md") for path in paths)
