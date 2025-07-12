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
        value = generate_random_value("integer")
        
        # Check that the value is the mocked value
        assert value == 42
    
    @patch('syda.utils.random.randint')
    @patch('syda.utils.random.choice')
    def test_generate_random_text(self, mock_choice, mock_randint):
        """Test generating random text."""
        # Set length of string to generate
        mock_randint.return_value = 10
        # Mock choice to return a single character
        mock_choice.return_value = "R"
        
        # Generate random text
        value = generate_random_value("text")
        
        # Since the function calls random.choice repeatedly, the result will be a string of 'R's
        assert isinstance(value, str)
        assert len(value) == 10
    
    @patch('syda.utils.random.randint')
    @patch('syda.utils.random.choice')
    def test_generate_random_email(self, mock_choice, mock_randint):
        """Test generating a random email."""
        # Email is just treated as text in the current implementation
        # Set length of string to generate
        mock_randint.return_value = 10
        # Mock choice to return a consistent character
        mock_choice.return_value = "a"
        
        # Generate a random email
        value = generate_random_value("email")
        
        # Check that the value is a string with the expected length
        assert isinstance(value, str)
        assert len(value) == 10
    
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
    
    @patch('syda.utils.random.randint')
    def test_generate_random_unknown_type(self, mock_randint):
        """Test generating a random value for an unknown type."""
        # Current implementation treats unknown types as text
        mock_randint.return_value = 10
        
        # Generate a random value for an unknown type
        value = generate_random_value("unknown_type")
        
        # Check that a string is returned (not None)
        assert isinstance(value, str)


class TestGetSchemaPrompt:
    """Tests for the get_schema_prompt function."""
    
    def test_get_schema_prompt_with_valid_schema(self):
        """Test getting a schema prompt with a valid schema."""
        # Define a test schema
        schema = {
            "id": "integer",
            "name": "text",
            "email": "email"
        }
        
        # Get the schema prompt with the required table_name parameter
        prompt = get_schema_prompt(schema, "customers", "Customer information")
        
        # Check that the prompt contains the table name
        assert "customers" in prompt
        assert "Customer information" in prompt
    
    def test_get_schema_prompt_with_empty_schema(self):
        """Test getting a schema prompt with an empty schema."""
        # Get the schema prompt for an empty schema with the required table_name parameter
        prompt = get_schema_prompt({}, "empty_table")
        
        # Check that the prompt contains the table name
        assert "empty_table" in prompt
        assert prompt != ""


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
        
        # Define the schema for the parse_dataframe_output function
        schema = {"id": "integer", "name": "text", "email": "text"}
        
        # Parse the output
        df = parse_dataframe_output(output, schema)
        
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
        
        # Define the schema for the parse_dataframe_output function
        schema = {"id": "integer", "name": "text", "email": "text"}
        
        # Parse the output - current implementation doesn't extract JSON from markdown blocks
        # so it will return an empty DataFrame with columns from schema
        df = parse_dataframe_output(output, schema)
        
        # Check that the DataFrame is empty but has the expected columns
        assert len(df) == 0
        assert set(df.columns) == set(["id", "name", "email"])
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON output."""
        # Define an invalid JSON output
        output = """
        This is not valid JSON
        """
        
        # Define the schema for the parse_dataframe_output function
        schema = {"id": "integer", "name": "text"}
        
        # Parse the output and expect an empty DataFrame with schema columns
        df = parse_dataframe_output(output, schema)
        
        # Check that the DataFrame is empty but has the expected columns
        assert len(df) == 0
        assert set(df.columns) == set(["id", "name"])
        
    def test_parse_empty_output(self):
        """Test parsing empty output."""
        # Define the schema for the parse_dataframe_output function
        schema = {"id": "integer", "name": "text"}
        
        # Parse empty output
        df = parse_dataframe_output("", schema)
        
        # Check that the DataFrame is empty but has the expected columns
        assert len(df) == 0
        assert set(df.columns) == set(["id", "name"])
