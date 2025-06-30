"""
Tests for the templates module.
"""
import pytest
from unittest.mock import patch, mock_open

from syda.templates import (
    get_template,
    get_prompt,
    replace_placeholders
)


class TestGetTemplate:
    """Tests for the get_template function."""
    
    @patch("builtins.open", new_callable=mock_open, read_data="This is the template content")
    def test_get_template_file(self, mock_file):
        """Test getting a template from a file."""
        # Get the template
        template = get_template("test_template.txt")
        
        # Check that the file was opened
        mock_file.assert_called_once_with("test_template.txt", "r")
        
        # Check that the template content was returned
        assert template == "This is the template content"
    
    def test_get_template_string(self):
        """Test getting a template from a string."""
        # Get the template
        template = get_template("This is a template string", is_file=False)
        
        # Check that the string was returned as is
        assert template == "This is a template string"
    
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_get_template_file_not_found(self, mock_file):
        """Test error handling when template file is not found."""
        # Try to get a non-existent template
        with pytest.raises(FileNotFoundError):
            get_template("nonexistent_template.txt")


class TestReplaceplaceholders:
    """Tests for the replace_placeholders function."""
    
    def test_replace_placeholders(self):
        """Test replacing placeholders in a template."""
        # Define a template with placeholders
        template = "Hello {name}, welcome to {place}!"
        
        # Define replacements
        replacements = {
            "name": "Alice",
            "place": "Wonderland"
        }
        
        # Replace placeholders
        result = replace_placeholders(template, replacements)
        
        # Check that placeholders were replaced
        assert result == "Hello Alice, welcome to Wonderland!"
    
    def test_replace_placeholders_missing_replacement(self):
        """Test error handling when a replacement is missing."""
        # Define a template with placeholders
        template = "Hello {name}, welcome to {place}!"
        
        # Define incomplete replacements
        replacements = {
            "name": "Alice"
        }
        
        # Replace placeholders
        with pytest.raises(KeyError, match="place"):
            replace_placeholders(template, replacements)
    
    def test_replace_placeholders_no_replacements(self):
        """Test replacing placeholders with no replacements needed."""
        # Define a template without placeholders
        template = "Hello world!"
        
        # Replace placeholders
        result = replace_placeholders(template, {})
        
        # Check that the template was returned unchanged
        assert result == "Hello world!"


class TestGetPrompt:
    """Tests for the get_prompt function."""
    
    @patch("syda.templates.get_template")
    def test_get_prompt_with_template_file(self, mock_get_template):
        """Test getting a prompt using a template file."""
        # Configure the mock
        mock_get_template.return_value = "Generate data for {schema_type} with fields: {schema_prompt}"
        
        # Define replacements
        replacements = {
            "schema_type": "Customer",
            "schema_prompt": "id:number, name:text, email:email"
        }
        
        # Get the prompt
        prompt = get_prompt("template.txt", replacements)
        
        # Check that get_template was called
        mock_get_template.assert_called_once_with("template.txt", is_file=True)
        
        # Check that the prompt was generated
        assert prompt == "Generate data for Customer with fields: id:number, name:text, email:email"
    
    @patch("syda.templates.get_template")
    def test_get_prompt_with_template_string(self, mock_get_template):
        """Test getting a prompt using a template string."""
        # Configure the mock
        mock_get_template.return_value = "Generate {count} rows for {schema_type}"
        
        # Define replacements
        replacements = {
            "count": "10",
            "schema_type": "Order"
        }
        
        # Get the prompt
        prompt = get_prompt("Generate {count} rows for {schema_type}", replacements, is_file=False)
        
        # Check that get_template was called
        mock_get_template.assert_called_once_with(
            "Generate {count} rows for {schema_type}", is_file=False
        )
        
        # Check that the prompt was generated
        assert prompt == "Generate 10 rows for Order"
    
    @patch("syda.templates.get_template")
    @patch("syda.templates.replace_placeholders")
    def test_get_prompt_calls_replace_placeholders(self, mock_replace, mock_get_template):
        """Test that get_prompt calls replace_placeholders."""
        # Configure the mocks
        mock_get_template.return_value = "template content"
        mock_replace.return_value = "replaced content"
        
        # Define replacements
        replacements = {"key": "value"}
        
        # Get the prompt
        prompt = get_prompt("template.txt", replacements)
        
        # Check that replace_placeholders was called
        mock_replace.assert_called_once_with("template content", replacements)
        
        # Check that the replaced content was returned
        assert prompt == "replaced content"
    
    @patch("syda.templates.get_template")
    def test_get_prompt_custom_provider_template(self, mock_get_template):
        """Test getting a provider-specific template."""
        # Define a provider-specific template dictionary
        templates = {
            "openai": "Generate data using the OpenAI model {model}",
            "anthropic": "Generate data using the Anthropic model {model}"
        }
        
        # Configure the mock to return the template dictionary
        mock_get_template.return_value = templates
        
        # Get the prompt for OpenAI
        prompt = get_prompt("templates.json", {"model": "gpt-4"}, provider="openai")
        
        # Check that the correct template was used
        assert prompt == "Generate data using the OpenAI model gpt-4"
        
        # Reset the mock
        mock_get_template.reset_mock()
        mock_get_template.return_value = templates
        
        # Get the prompt for Anthropic
        prompt = get_prompt("templates.json", {"model": "claude-3"}, provider="anthropic")
        
        # Check that the correct template was used
        assert prompt == "Generate data using the Anthropic model claude-3"
    
    @patch("syda.templates.get_template")
    def test_get_prompt_provider_template_not_found(self, mock_get_template):
        """Test error handling when a provider template is not found."""
        # Define a provider-specific template dictionary without the requested provider
        templates = {
            "openai": "OpenAI template"
        }
        
        # Configure the mock to return the template dictionary
        mock_get_template.return_value = templates
        
        # Try to get a prompt for a non-existent provider
        with pytest.raises(ValueError, match="No template found for provider"):
            get_prompt("templates.json", {}, provider="unknown")
