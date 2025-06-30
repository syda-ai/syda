"""
Tests for the schema_loader module.
"""
import pytest
import tempfile
import os
import json
import yaml
from unittest.mock import patch, mock_open, MagicMock

from syda.schema_loader import SchemaLoader


@pytest.fixture
def sample_schema_dict():
    """Return a sample schema dictionary."""
    return {
        "id": {"type": "number", "description": "Unique identifier"},
        "name": {"type": "text", "description": "Customer name"},
        "email": {"type": "email", "description": "Customer email"},
        "__metadata__": {"description": "Customer schema"}
    }


@pytest.fixture
def sample_schema_with_fk():
    """Return a sample schema with foreign key."""
    return {
        "id": {"type": "number", "description": "Unique identifier"},
        "customer_id": {
            "type": "foreign_key", 
            "description": "Reference to customer", 
            "references": {"schema": "Customer", "field": "id"}
        },
        "__foreign_keys__": {
            "customer_id": ["Customer", "id"]
        }
    }


class TestSchemaLoader:
    """Tests for the SchemaLoader class."""
    
    def test_load_schema_dict(self, sample_schema_dict):
        """Test loading schema from a dictionary."""
        loader = SchemaLoader()
        
        schema, metadata, desc, fks, template, depends_on = loader.load_schema(sample_schema_dict)
        
        # Check the schema is properly processed
        assert "id" in schema
        assert schema["id"]["type"] == "number"
        assert metadata == {"description": "Customer schema"}
        assert desc == "Customer schema"
        assert fks == {}  # No foreign keys
        assert template == {}  # No templates
        assert depends_on == []  # No dependencies
        
    def test_load_schema_with_foreign_keys(self, sample_schema_with_fk):
        """Test loading schema with foreign keys."""
        loader = SchemaLoader()
        
        schema, metadata, desc, fks, template, depends_on = loader.load_schema(sample_schema_with_fk)
        
        # Check that foreign keys were extracted
        assert "customer_id" in fks
        assert fks["customer_id"] == ["Customer", "id"]
        
        # Check dependencies were identified
        assert "Customer" in depends_on
        
    def test_load_schema_json_file(self, sample_schema_dict):
        """Test loading schema from a JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp.write(json.dumps(sample_schema_dict).encode('utf-8'))
            tmp_path = tmp.name
        
        try:
            # Load the schema from the file
            loader = SchemaLoader()
            schema, metadata, desc, fks, template, depends_on = loader.load_schema(tmp_path)
            
            # Check the schema is properly loaded
            assert "id" in schema
            assert schema["id"]["type"] == "number"
            assert metadata == {"description": "Customer schema"}
        finally:
            # Clean up
            os.unlink(tmp_path)
            
    def test_load_schema_yaml_file(self, sample_schema_dict):
        """Test loading schema from a YAML file."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            tmp.write(yaml.dump(sample_schema_dict).encode('utf-8'))
            tmp_path = tmp.name
        
        try:
            # Load the schema from the file
            loader = SchemaLoader()
            schema, metadata, desc, fks, template, depends_on = loader.load_schema(tmp_path)
            
            # Check the schema is properly loaded
            assert "id" in schema
            assert schema["id"]["type"] == "number"
            assert metadata == {"description": "Customer schema"}
        finally:
            # Clean up
            os.unlink(tmp_path)
            
    def test_load_schema_unsupported_file_type(self):
        """Test error handling for unsupported file type."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"Not a schema file")
            tmp_path = tmp.name
        
        try:
            # Try to load the schema
            loader = SchemaLoader()
            with pytest.raises(ValueError, match="Unsupported file type"):
                loader.load_schema(tmp_path)
        finally:
            # Clean up
            os.unlink(tmp_path)
            
    @patch('builtins.open', new_callable=mock_open)
    def test_load_schema_file_not_found(self, mock_file):
        """Test error handling when file is not found."""
        # Mock the open function to raise FileNotFoundError
        mock_file.side_effect = FileNotFoundError
        
        loader = SchemaLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_schema("nonexistent_file.json")
            
    def test_load_schema_sqlalchemy_model(self):
        """Test loading schema from SQLAlchemy model."""
        # Create a mock SQLAlchemy model
        mock_model = MagicMock()
        mock_model.__tablename__ = "TestModel"
        
        # Mock the SQLAlchemy inspection
        with patch('syda.schema_loader.DeclarativeMeta', object):
            with patch('syda.schema_loader.inspect') as mock_inspect:
                # Configure the mock inspector
                mock_inspector = MagicMock()
                mock_inspect.return_value = mock_inspector
                
                # Configure columns for the mock
                column1 = MagicMock()
                column1.name = "id"
                column1.type.python_type = int
                column1.primary_key = True
                
                column2 = MagicMock()
                column2.name = "name"
                column2.type.python_type = str
                
                mock_inspector.columns = [column1, column2]
                
                # Mock no relationships
                mock_inspector.relationships = []
                
                # Load the schema from the mock model
                loader = SchemaLoader()
                schema, metadata, desc, fks, template, depends_on = loader.load_schema(mock_model)
                
                # Check the schema is properly converted
                assert "id" in schema
                assert schema["id"]["type"] == "number"
                assert "name" in schema
                assert schema["name"]["type"] == "text"
                
    def test_process_schema_for_llm(self):
        """Test processing schema for LLM."""
        raw_schema = {
            "id": "number",
            "name": "text",
            "active": True  # Non-standard format
        }
        
        loader = SchemaLoader()
        processed = loader._process_schema_for_llm(raw_schema)
        
        # Check the processed schema
        assert processed["id"]["type"] == "number"
        assert processed["name"]["type"] == "text"
        assert processed["active"]["type"] == "boolean"
        
    def test_extract_foreign_keys(self):
        """Test extraction of foreign keys from schema."""
        schema = {
            "customer_id": {
                "type": "foreign_key", 
                "references": {"schema": "Customer", "field": "id"}
            },
            "product_id": {
                "type": "foreign_key",
                "references": "Product.id"  # Alternative format
            }
        }
        
        loader = SchemaLoader()
        fks = loader._extract_foreign_keys(schema)
        
        # Check the extracted foreign keys
        assert "customer_id" in fks
        assert fks["customer_id"] == ("Customer", "id")
        assert "product_id" in fks
        assert fks["product_id"] == ("Product", "id")
