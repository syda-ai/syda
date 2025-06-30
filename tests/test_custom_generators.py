"""
Tests for the custom_generators module.
"""
import pytest
import random
from unittest.mock import patch, MagicMock

from syda.custom_generators import GeneratorManager


class TestGeneratorManager:
    """Tests for the GeneratorManager class."""
    
    def test_initialization(self):
        """Test initialization of GeneratorManager."""
        manager = GeneratorManager()
        
        # Check that attributes are properly initialized
        assert hasattr(manager, "type_generators")
        assert hasattr(manager, "column_generators")
        assert isinstance(manager.type_generators, dict)
        assert isinstance(manager.column_generators, dict)
    
    def test_register_type_generator(self):
        """Test registering a type generator."""
        manager = GeneratorManager()
        
        # Define a test generator
        def test_generator(schema_item):
            return "test_value"
        
        # Register the generator
        manager.register_type_generator("test_type", test_generator)
        
        # Check that the generator was registered
        assert "test_type" in manager.type_generators
        assert manager.type_generators["test_type"] == test_generator
    
    def test_register_column_generator(self):
        """Test registering a column generator."""
        manager = GeneratorManager()
        
        # Define a test generator
        def test_generator(schema_item):
            return "test_value"
        
        # Register the generator
        manager.register_column_generator("test_column", test_generator)
        
        # Check that the generator was registered
        assert "test_column" in manager.column_generators
        assert manager.column_generators["test_column"] == test_generator
    
    def test_register_generators_from_dict(self):
        """Test registering generators from a dictionary."""
        manager = GeneratorManager()
        
        # Define test generators
        def type_gen(schema_item):
            return "type_value"
        
        def column_gen(schema_item):
            return "column_value"
        
        generators = {
            "types": {
                "custom_type": type_gen
            },
            "columns": {
                "custom_column": column_gen
            }
        }
        
        # Register the generators
        manager.register_generators_from_dict(generators)
        
        # Check that the generators were registered
        assert "custom_type" in manager.type_generators
        assert manager.type_generators["custom_type"] == type_gen
        assert "custom_column" in manager.column_generators
        assert manager.column_generators["custom_column"] == column_gen
    
    def test_get_generator_column_precedence(self):
        """Test generator precedence with column generator."""
        manager = GeneratorManager()
        
        # Define test generators
        def type_gen(schema_item):
            return "type_value"
        
        def column_gen(schema_item):
            return "column_value"
        
        # Register the generators
        manager.register_type_generator("test_type", type_gen)
        manager.register_column_generator("test_column", column_gen)
        
        # Get the generator for the column (column generator should have precedence)
        generator = manager.get_generator("test_column", {"type": "test_type"})
        
        # Check that the column generator was returned
        assert generator == column_gen
    
    def test_get_generator_type_fallback(self):
        """Test generator fallback to type generator."""
        manager = GeneratorManager()
        
        # Define a test generator
        def type_gen(schema_item):
            return "type_value"
        
        # Register the generator
        manager.register_type_generator("test_type", type_gen)
        
        # Get the generator for a column with no specific generator
        generator = manager.get_generator("some_column", {"type": "test_type"})
        
        # Check that the type generator was returned
        assert generator == type_gen
    
    def test_get_generator_missing(self):
        """Test behavior when no generator is found."""
        manager = GeneratorManager()
        
        # Get the generator for a column with no registered generator
        generator = manager.get_generator("some_column", {"type": "unknown_type"})
        
        # Check that None was returned
        assert generator is None
    
    def test_generate_with_registered_generator(self):
        """Test generating a value using a registered generator."""
        manager = GeneratorManager()
        
        # Define a test generator
        def test_generator(schema_item):
            return "generated_value"
        
        # Register the generator
        manager.register_type_generator("test_type", test_generator)
        
        # Generate a value
        value = manager.generate("test_column", {"type": "test_type"}, {})
        
        # Check that the generator was used
        assert value == "generated_value"
    
    @patch("syda.utils.generate_random_value")
    def test_generate_fallback_random(self, mock_generate_random):
        """Test fallback to random value generation."""
        manager = GeneratorManager()
        
        # Configure the mock
        mock_generate_random.return_value = "random_value"
        
        # Generate a value with no registered generator
        value = manager.generate("test_column", {"type": "unknown_type"}, {})
        
        # Check that random value generation was used
        mock_generate_random.assert_called_once_with("unknown_type")
        assert value == "random_value"
    
    def test_generate_text(self):
        """Test generating text values."""
        manager = GeneratorManager()
        
        # Create schema item with min_length and max_length
        schema_item = {
            "type": "text",
            "min_length": 5,
            "max_length": 10
        }
        
        # Generate a text value
        value = manager.generate_text(schema_item)
        
        # Check that the value is a string of the expected length
        assert isinstance(value, str)
        assert 5 <= len(value) <= 10
    
    def test_generate_email(self):
        """Test generating email values."""
        manager = GeneratorManager()
        
        # Generate an email value
        value = manager.generate_email({})
        
        # Check that the value is a string with @ and a domain
        assert isinstance(value, str)
        assert "@" in value
        assert "." in value.split("@")[1]
    
    def test_generate_date(self):
        """Test generating date values."""
        manager = GeneratorManager()
        
        # Generate a date value
        value = manager.generate_date({})
        
        # Check that the value is a string in YYYY-MM-DD format
        assert isinstance(value, str)
        parts = value.split("-")
        assert len(parts) == 3
        year, month, day = map(int, parts)
        assert 1900 <= year <= 2100
        assert 1 <= month <= 12
        assert 1 <= day <= 31
    
    def test_generate_boolean(self):
        """Test generating boolean values."""
        manager = GeneratorManager()
        
        # Set a random seed for deterministic testing
        random.seed(42)
        
        # Generate a boolean value
        value = manager.generate_boolean({})
        
        # Check that the value is a boolean
        assert isinstance(value, bool)
    
    def test_generate_number(self):
        """Test generating number values."""
        manager = GeneratorManager()
        
        # Create schema item with min and max
        schema_item = {
            "type": "number",
            "min": 5,
            "max": 10
        }
        
        # Generate a number value
        value = manager.generate_number(schema_item)
        
        # Check that the value is a number in the expected range
        assert isinstance(value, (int, float))
        assert 5 <= value <= 10
    
    def test_generate_number_with_precision(self):
        """Test generating number values with precision."""
        manager = GeneratorManager()
        
        # Create schema item with precision
        schema_item = {
            "type": "number",
            "min": 5,
            "max": 10,
            "precision": 2
        }
        
        # Generate a number value
        value = manager.generate_number(schema_item)
        
        # Check that the value is a float with at most 2 decimal places
        assert isinstance(value, float)
        assert 5 <= value <= 10
        assert abs(value - round(value, 2)) < 1e-10
