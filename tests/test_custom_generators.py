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
        def test_generator(row, col_name):
            return "test_value"
        
        # Register the generator
        manager.register_generator("test_type", test_generator)
        
        # Check that the generator was registered
        assert "test_type" in manager.type_generators
        assert manager.type_generators["test_type"] == test_generator
    
    def test_register_column_generator(self):
        """Test registering a column generator."""
        manager = GeneratorManager()
        
        # Define a test generator
        def test_generator(row, col_name):
            return "test_value"
        
        # Register the generator with a column name
        manager.register_generator("test_type", test_generator, column_name="test_column")
        
        # Check that the generator was registered
        assert "test_column" in manager.column_generators
        assert manager.column_generators["test_column"] == test_generator
    
    def test_get_generator_state(self):
        """Test getting and restoring generator state."""
        manager = GeneratorManager()
        
        # Define test generators
        def type_gen(row, col_name):
            return "type_value"
        
        def column_gen(row, col_name):
            return "column_value"
        
        # Register the generators
        manager.register_generator("custom_type", type_gen)
        manager.register_generator("custom_type2", column_gen, column_name="custom_column")
        
        # Get the state
        state = manager.get_generator_state()
        
        # Check that the state was returned correctly
        type_generators, column_generators = state
        assert "custom_type" in type_generators
        assert type_generators["custom_type"] == type_gen
        assert "custom_column" in column_generators
        assert column_generators["custom_column"] == column_gen
    
    def test_type_and_column_generators(self):
        """Test that type and column generators are stored in their respective dictionaries."""
        manager = GeneratorManager()
        
        # Define test generators
        def type_gen(row, col_name):
            return "type_value"
        
        def column_gen(row, col_name):
            return "column_value"
        
        # Register the generators
        manager.register_generator("test_type", type_gen)
        manager.register_generator("test_type", column_gen, column_name="test_column")
        
        # Check that the generators were registered in their respective dictionaries
        assert "test_type" in manager.type_generators
        assert manager.type_generators["test_type"] == type_gen
        assert "test_column" in manager.column_generators
        assert manager.column_generators["test_column"] == column_gen
    
    def test_restore_generator_state(self):
        """Test saving and restoring generator state."""
        manager = GeneratorManager()
        
        # Define a test generator
        def type_gen(row, col_name):
            return "type_value"
        
        # Register the generator
        manager.register_generator("test_type", type_gen)
        
        # Get the state
        state = manager.get_generator_state()
        
        # Create a new manager
        new_manager = GeneratorManager()
        
        # Restore the state
        new_manager.restore_generator_state(state)
        
        # Check that the generator was restored
        assert "test_type" in new_manager.type_generators
        assert new_manager.type_generators["test_type"] == type_gen
    

    

    

    
    # Additional tests for GeneratorManager functionality could be added here
    
    # The following tests have been removed because they reference methods that don't exist
    # in the actual GeneratorManager implementation:
    # - test_generate_boolean
    # - test_generate_number
    # - test_generate_number_with_precision
