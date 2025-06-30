"""
Tests for the utils module.
"""
import pytest
import pandas as pd
import random
from unittest.mock import patch

from syda.utils import (
    create_empty_dataframe,
    generate_random_value,
    get_schema_prompt,
    parse_dataframe_output
)


class TestCreateEmptyDataframe:
    """Tests for the create_empty_dataframe function."""
    
    def test_create_empty_dataframe_with_valid_schema(self):
        """Test creating an empty DataFrame with a valid schema."""
        # Define a test schema
        schema = {
            "id": {"type": "number", "description": "Unique identifier"},
            "name": {"type": "text", "description": "Customer name"},
            "email": {"type": "email", "description": "Customer email"}
        }
        
        # Create an empty DataFrame
        df = create_empty_dataframe(schema)
        
        # Check that the DataFrame has the expected columns
        assert list(df.columns) == ["id", "name", "email"]
        assert len(df) == 0
    
    def test_create_empty_dataframe_with_empty_schema(self):
        """Test creating an empty DataFrame with an empty schema."""
        # Create an empty DataFrame with an empty schema
        df = create_empty_dataframe({})
        
        # Check that the DataFrame is empty
        assert len(df.columns) == 0
        assert len(df) == 0


class TestGenerateRandomValue:
    """Tests for the generate_random_value function."""
    
    @patch('syda.utils.random.randint')
    def test_generate_random_number(self, mock_randint):
        """Test generating a random number."""
        mock_randint.return_value = 42
        
        # Generate a random number
        value = generate_random_value("number")
        
        # Check that the value is the mocked value
        assert value == 42
    
    @patch('syda.utils.random.choice')
    def test_generate_random_text(self, mock_choice):
        """Test generating random text."""
        mock_choice.return_value = "RandomText"
        
        # Generate random text
        value = generate_random_value("text")
        
        # Check that the value is the mocked value
        assert value == "RandomText"
    
    @patch('syda.utils.random.choice')
    def test_generate_random_email(self, mock_choice):
        """Test generating a random email."""
        # Set up the mock to return predefined values
        mock_choice.side_effect = ["john", "example.com"]
        
        # Generate a random email
        value = generate_random_value("email")
        
        # Check that the value is in the expected format
        assert "@" in value
        assert value.endswith(".com")
    
    def test_generate_random_boolean(self):
        """Test generating a random boolean."""
        # Seed random for deterministic results
        random.seed(42)
        
        # Generate a random boolean
        value = generate_random_value("boolean")
        
        # Check that the value is a boolean
        assert isinstance(value, bool)
    
    def test_generate_random_date(self):
        """Test generating a random date."""
        # Generate a random date
        value = generate_random_value("date")
        
        # Check that the value is in the expected format (YYYY-MM-DD)
        assert len(value.split("-")) == 3
        year, month, day = map(int, value.split("-"))
        assert 1900 <= year <= 2100
        assert 1 <= month <= 12
        assert 1 <= day <= 31
    
    def test_generate_random_unknown_type(self):
        """Test generating a random value for an unknown type."""
        # Generate a random value for an unknown type
        value = generate_random_value("unknown_type")
        
        # Check that the value is None
        assert value is None


class TestGetSchemaPrompt:
    """Tests for the get_schema_prompt function."""
    
    def test_get_schema_prompt_with_valid_schema(self):
        """Test getting a schema prompt with a valid schema."""
        # Define a test schema
        schema = {
            "id": {"type": "number", "description": "Unique identifier"},
            "name": {"type": "text", "description": "Customer name"},
            "email": {"type": "email", "description": "Customer email"}
        }
        
        # Get the schema prompt
        prompt = get_schema_prompt(schema)
        
        # Check that the prompt contains the expected information
        assert "id" in prompt
        assert "number" in prompt
        assert "Unique identifier" in prompt
        assert "name" in prompt
        assert "text" in prompt
        assert "Customer name" in prompt
        assert "email" in prompt
        assert "email" in prompt  # Type and field name are the same
        assert "Customer email" in prompt
    
    def test_get_schema_prompt_with_empty_schema(self):
        """Test getting a schema prompt with an empty schema."""
        # Get the schema prompt for an empty schema
        prompt = get_schema_prompt({})
        
        # Check that the prompt is not empty, but is minimal
        assert prompt != ""
        assert "{}" in prompt


class TestParseDataframeOutput:
    """Tests for the parse_dataframe_output function."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON output."""
        # Define a valid JSON output
        output = """
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ]
        """
        
        # Parse the output
        df = parse_dataframe_output(output)
        
        # Check that the DataFrame has the expected data
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "email"]
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[1]["email"] == "bob@example.com"
    
    def test_parse_json_with_extra_text(self):
        """Test parsing JSON output with extra text."""
        # Define a JSON output with extra text
        output = """
        Here is the data:
        ```json
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ]
        ```
        """
        
        # Parse the output
        df = parse_dataframe_output(output)
        
        # Check that the DataFrame has the expected data
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "email"]
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON output."""
        # Define an invalid JSON output
        output = """
        This is not valid JSON
        """
        
        # Parse the output and expect an empty DataFrame
        df = parse_dataframe_output(output)
        
        # Check that the DataFrame is empty
        assert len(df) == 0
        assert len(df.columns) == 0
        
    def test_parse_empty_output(self):
        """Test parsing empty output."""
        # Parse empty output
        df = parse_dataframe_output("")
        
        # Check that the DataFrame is empty
        assert len(df) == 0
        assert len(df.columns) == 0
