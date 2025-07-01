"""
Tests for the templates module.
"""
import pytest
import os
import re
from unittest.mock import patch, mock_open, MagicMock

from syda.templates import TemplateProcessor


class TestTemplateProcessor:
    """Tests for the TemplateProcessor class."""
    
    @pytest.fixture
    def template_processor(self):
        """Create a TemplateProcessor instance for testing."""
        # Mock the UnstructuredDataProcessor to avoid circular imports
        with patch('syda.templates.UnstructuredDataProcessor'):
            processor = TemplateProcessor()
            return processor
    
    def test_initialization(self, template_processor):
        """Test TemplateProcessor initialization."""
        assert hasattr(template_processor, 'placeholder_pattern')
        assert isinstance(template_processor.placeholder_pattern, re.Pattern)
        
    def test_extract_placeholders(self, template_processor):
        """Test extracting placeholders from a template."""
        template = "Hello {{ name }}, your age is {{ age }}!"
        placeholders = template_processor.extract_placeholders(template)
        
        assert placeholders == {"name", "age"}
        
    def test_extract_placeholders_no_placeholders(self, template_processor):
        """Test extracting placeholders from a template with no placeholders."""
        template = "Hello world, no placeholders here!"
        placeholders = template_processor.extract_placeholders(template)
        
        assert placeholders == set()
        
    @patch("os.path.exists")
    def test_get_template_content(self, mock_exists, template_processor):
        """Test getting content from a template file."""
        # Configure mocks
        mock_exists.return_value = True
        template_processor.file_processor = MagicMock()
        template_processor.file_processor.process_file.return_value = {
            'text': 'Template content',
            'type': 'text'
        }
        
        # Get the template content
        content = template_processor.get_template_content("test_template.txt")
        
        # Check that the file processor was called
        template_processor.file_processor.process_file.assert_called_once_with("test_template.txt")
        
        # Check that the content was returned
        assert content == "Template content"
    
    @patch("os.path.exists")
    def test_get_template_content_file_not_found(self, mock_exists, template_processor):
        """Test error handling when template file is not found."""
        # Configure mock
        mock_exists.return_value = False
        
        # Try to get a non-existent template
        with pytest.raises(ValueError, match="Template file not found"):
            template_processor.get_template_content("nonexistent_template.txt")
            
    @patch("os.path.exists")
    def test_get_template_content_extraction_error(self, mock_exists, template_processor):
        """Test error handling when template extraction fails."""
        # Configure mocks
        mock_exists.return_value = True
        template_processor.file_processor = MagicMock()
        template_processor.file_processor.process_file.return_value = {
            'error': 'Failed to extract text',
            'type': 'text'
        }
        
        # Try to get a template with extraction error
        with pytest.raises(ValueError, match="Error processing template file"):
            template_processor.get_template_content("error_template.txt")
            
    def test_replace_placeholders(self, template_processor):
        """Test replacing placeholders in a template."""
        # Define a template with placeholders
        template = "Hello {{ name }}, welcome to {{ place }}!"
        
        # Define replacements
        replacements = {
            "name": "Alice",
            "place": "Wonderland"
        }
        
        # Replace placeholders
        result = template_processor.replace_placeholders(template, replacements)
        
        # Check that placeholders were replaced
        assert result == "Hello Alice, welcome to Wonderland!"
    
    def test_replace_placeholders_with_numbers(self, template_processor):
        """Test replacing placeholders with numeric values."""
        # Define a template with placeholders
        template = "The answer is {{ answer }}!"
        
        # Define replacements
        replacements = {"answer": 42}
        
        # Replace placeholders
        result = template_processor.replace_placeholders(template, replacements)
        
        # Check that placeholders were replaced with string representation
        assert result == "The answer is 42!"
        
    def test_create_schema_from_placeholders(self, template_processor):
        """Test creating a schema from placeholders."""
        # Define a set of placeholders
        placeholders = {"name", "email", "age", "birth_date"}
        
        # Create the schema
        schema = template_processor.create_schema_from_placeholders(placeholders)
        
        # Check that the schema has the expected fields
        assert "name" in schema
        assert "email" in schema
        assert "age" in schema
        assert "birth_date" in schema
        
    @patch('jinja2.Environment')
    def test_render_jinja2_template(self, mock_env, template_processor):
        """Test rendering a Jinja2 template."""
        # Mock the jinja2 environment and template
        mock_template = MagicMock()
        mock_template.render.return_value = "Hello Alice!"
        
        mock_env_instance = MagicMock()
        mock_env_instance.get_template.return_value = mock_template
        mock_env.return_value = mock_env_instance
        
        # Render the template
        result = template_processor.render_jinja2_template("template.j2", {"name": "Alice"})
        
        # Check that the template was rendered
        assert result == "Hello Alice!"
        mock_template.render.assert_called_once_with(name="Alice")
        
    @patch("os.path.exists")
    def test_process_template_with_data(self, mock_exists, template_processor):
        """Test processing a template with data."""
        # Configure mocks
        mock_exists.return_value = True
        
        # Mock methods
        template_processor.get_template_content = MagicMock(return_value="Hello {{ name }}!")
        template_processor.replace_placeholders = MagicMock(return_value="Hello Alice!")
        template_processor._process_text_template = MagicMock(return_value="output.txt")
        
        # Process the template
        result = template_processor.process_template_with_data(
            "template.txt",
            {"name": "Alice"},
            "output.txt",
            input_file_type="txt",
            output_file_type="txt"
        )
        
        # Check that the methods were called
        template_processor.get_template_content.assert_called_once_with("template.txt")
        template_processor.replace_placeholders.assert_called_once_with("Hello {{ name }}!", {"name": "Alice"})
        template_processor._process_text_template.assert_called_once_with("Hello Alice!", "output.txt", "txt")
        
        # Check that the result was returned
        assert result == "output.txt"
        
    def test_process_html_template(self, template_processor):
        """Test processing an HTML template."""
        # Mock file processing
        with patch("builtins.open", mock_open()) as mock_file:
            # Process the template
            result = template_processor._process_html_template("HTML content", "output.html", "html")
            
            # Check that the file was written
            mock_file.assert_called_once_with("output.html", "w")
            mock_file().write.assert_called_once_with("HTML content")
            
            # Check that the result was returned
            assert result == "output.html"
