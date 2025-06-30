"""
Tests for the generate module.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call

from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig


@pytest.fixture
def sample_schema():
    """Return a sample schema dictionary."""
    return {
        "id": {"type": "number", "description": "Unique identifier"},
        "name": {"type": "text", "description": "Customer name"},
        "email": {"type": "email", "description": "Customer email"}
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
        }
    }


@pytest.fixture
def model_config():
    """Return a sample model config."""
    return ModelConfig(provider="openai", model_name="gpt-4")


class TestSyntheticDataGenerator:
    """Tests for the SyntheticDataGenerator class."""
    
    def test_initialization(self, model_config):
        """Test initialization of SyntheticDataGenerator."""
        generator = SyntheticDataGenerator(model_config=model_config, api_key="test_key")
        
        # Check that attributes are properly initialized
        assert generator.model_config == model_config
        assert generator.api_key == "test_key"
        assert generator.schemas == {}
        assert generator.dataframes == {}
    
    @patch("syda.generate.SchemaLoader")
    def test_load_schema(self, mock_schema_loader, sample_schema):
        """Test loading a schema."""
        # Configure the mock
        mock_loader_instance = MagicMock()
        mock_schema_loader.return_value = mock_loader_instance
        mock_loader_instance.load_schema.return_value = (
            sample_schema, {}, "Test schema", {}, {}, []
        )
        
        # Create the generator
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        
        # Load the schema
        generator.load_schema("customers", sample_schema)
        
        # Check that the schema was loaded
        mock_loader_instance.load_schema.assert_called_once_with(sample_schema)
        assert "customers" in generator.schemas
        assert generator.schemas["customers"]["schema"] == sample_schema
        
    @patch("syda.generate.DependencyHandler")
    @patch("syda.generate.SchemaLoader")
    def test_load_schema_with_dependencies(self, mock_schema_loader, mock_dependency_handler, sample_schema_with_fk):
        """Test loading a schema with dependencies."""
        # Configure the schema loader mock
        mock_loader_instance = MagicMock()
        mock_schema_loader.return_value = mock_loader_instance
        mock_loader_instance.load_schema.return_value = (
            sample_schema_with_fk, {}, "Test schema", 
            {"customer_id": ("Customer", "id")},  # Foreign keys
            {}, ["Customer"]  # Dependencies
        )
        
        # Create the generator
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        
        # Load the schema
        generator.load_schema("orders", sample_schema_with_fk)
        
        # Check that the schema was loaded with foreign keys
        assert "orders" in generator.schemas
        assert generator.schemas["orders"]["foreign_keys"] == {"customer_id": ("Customer", "id")}
        
        # Check that dependency extraction was called
        assert mock_dependency_handler.extract_dependencies.called
    
    @patch("syda.generate.create_llm_client")
    def test_generate_data(self, mock_create_client, sample_schema):
        """Test generating data."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        # Mock response from the LLM
        mock_response = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "email": ["alice@example.com", "bob@example.com"]
        })
        mock_client.generate_synthetic_data.return_value = mock_response
        
        # Create the generator with a loaded schema
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        generator.schemas = {
            "customers": {
                "schema": sample_schema,
                "description": "Test schema",
                "foreign_keys": {}
            }
        }
        
        # Generate data
        df = generator.generate("customers", 2)
        
        # Check that the client was called
        mock_create_client.assert_called_once()
        mock_client.generate_synthetic_data.assert_called_once()
        
        # Check that the DataFrame was returned
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        
    def test_generate_data_for_unknown_schema(self):
        """Test generating data for an unknown schema."""
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        
        # Try to generate data for an unknown schema
        with pytest.raises(ValueError, match="Schema .* not found"):
            generator.generate("unknown_schema", 10)
    
    @patch("syda.generate.ForeignKeyHandler")
    @patch("syda.generate.create_llm_client")
    def test_generate_data_with_foreign_keys(self, mock_create_client, mock_fk_handler, sample_schema_with_fk):
        """Test generating data with foreign keys."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        # Mock response from the LLM
        mock_response = pd.DataFrame({
            "id": [1, 2],
            "customer_id": [None, None]  # Foreign keys to be resolved
        })
        mock_client.generate_synthetic_data.return_value = mock_response
        
        # Configure the foreign key handler mock
        mock_handler_instance = MagicMock()
        mock_fk_handler.return_value = mock_handler_instance
        
        # Create the generator with loaded schemas
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        generator.schemas = {
            "customers": {
                "schema": {"id": {"type": "number"}},
                "description": "Customer schema",
                "foreign_keys": {}
            },
            "orders": {
                "schema": sample_schema_with_fk,
                "description": "Order schema",
                "foreign_keys": {"customer_id": ("Customer", "id")}
            }
        }
        generator.dataframes = {
            "customers": pd.DataFrame({"id": [101, 102]})
        }
        
        # Generate data
        generator.generate("orders", 2)
        
        # Check that the foreign key handler was used
        mock_handler_instance.register_dataframes.assert_called_once_with(generator.dataframes)
        mock_handler_instance.resolve_foreign_key_dependencies.assert_called()
    
    @patch("syda.generate.create_empty_dataframe")
    @patch("syda.generate.GeneratorManager")
    def test_generate_random_data(self, mock_generator_manager, mock_create_empty_df, sample_schema):
        """Test generating random data."""
        # Configure the empty DataFrame mock
        mock_df = pd.DataFrame(columns=["id", "name", "email"])
        mock_create_empty_df.return_value = mock_df
        
        # Configure the generator manager mock
        mock_manager_instance = MagicMock()
        mock_generator_manager.return_value = mock_manager_instance
        mock_manager_instance.generate.side_effect = [1, "Alice", "alice@example.com"]
        
        # Create the generator with a loaded schema
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        generator.schemas = {
            "customers": {
                "schema": sample_schema,
                "description": "Test schema",
                "foreign_keys": {}
            }
        }
        
        # Generate random data
        df = generator.generate_random("customers", 1)
        
        # Check that the manager was called for each field
        assert mock_manager_instance.generate.call_count == 3
        mock_manager_instance.generate.assert_has_calls([
            call("id", sample_schema["id"], {}),
            call("name", sample_schema["name"], {}),
            call("email", sample_schema["email"], {})
        ], any_order=True)
        
        # Check that the DataFrame was returned
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    
    @patch("syda.generate.save_dataframes")
    def test_save_dataframes(self, mock_save_dataframes):
        """Test saving DataFrames."""
        # Create DataFrames
        df1 = pd.DataFrame({"id": [1, 2]})
        df2 = pd.DataFrame({"id": [3, 4]})
        
        # Create the generator with DataFrames
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        generator.dataframes = {
            "table1": df1,
            "table2": df2
        }
        
        # Save the DataFrames
        generator.save("output_dir", format="csv")
        
        # Check that save_dataframes was called
        mock_save_dataframes.assert_called_once_with(
            generator.dataframes, "output_dir", format="csv", filenames=None
        )
    
    @patch("syda.generate.DependencyHandler")
    def test_generate_all(self, mock_dependency_handler):
        """Test generating all data."""
        # Configure the dependency handler mock
        mock_dependency_handler.build_dependency_graph.return_value = MagicMock()
        mock_dependency_handler.determine_generation_order.return_value = ["table1", "table2"]
        
        # Create the generator with schemas
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        generator.schemas = {
            "table1": {"schema": {}, "description": "Table 1"},
            "table2": {"schema": {}, "description": "Table 2"}
        }
        
        # Mock the generate method
        generator.generate = MagicMock()
        
        # Generate all data
        generator.generate_all(10)
        
        # Check that generate was called for each table
        assert generator.generate.call_count == 2
        generator.generate.assert_has_calls([
            call("table1", 10),
            call("table2", 10)
        ])
        
    @patch("syda.generate.DependencyHandler")
    def test_detect_dependency_cycle(self, mock_dependency_handler):
        """Test detecting a dependency cycle."""
        # Configure the dependency handler mock
        mock_dependency_handler.build_dependency_graph.return_value = MagicMock()
        mock_dependency_handler.has_cycle.return_value = True
        
        # Create the generator with schemas
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        
        # Check for cycles
        with pytest.raises(ValueError, match="Circular dependencies detected"):
            generator._check_for_dependency_cycles()
            
    @patch("syda.generate.create_llm_client")
    def test_client_error_handling(self, mock_create_client):
        """Test error handling when the client fails."""
        # Configure the LLM client mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.generate_synthetic_data.side_effect = Exception("API Error")
        
        # Create the generator with a loaded schema
        generator = SyntheticDataGenerator(model_config=MagicMock(), api_key="test_key")
        generator.schemas = {
            "customers": {
                "schema": {"id": {"type": "number"}},
                "description": "Test schema",
                "foreign_keys": {}
            }
        }
        
        # Generate data with fallback
        with patch.object(generator, "generate_random") as mock_generate_random:
            mock_generate_random.return_value = pd.DataFrame({"id": [1, 2]})
            
            df = generator.generate("customers", 2)
            
            # Check that generate_random was called as fallback
            mock_generate_random.assert_called_once_with("customers", 2)
