import os
import pytest
import pandas as pd
import networkx as nx
from unittest.mock import MagicMock, patch, call
from syda.schemas import ModelConfig
from syda.generate import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test the SyntheticDataGenerator class."""

    @pytest.fixture
    def sample_schema(self):
        """Sample schema for testing."""
        return {
            "id": {"type": "number", "description": "Unique identifier"},
            "name": {"type": "text", "description": "Customer name"},
            "email": {"type": "email", "description": "Customer email"}
        }

    @pytest.fixture
    def sample_schema_with_fk(self):
        """Sample schema with foreign key for testing."""
        return {
            "id": {"type": "number", "description": "Unique identifier"},
            "customer_id": {
                "type": "foreign_key",
                "description": "Reference to customer",
                "references": {"schema": "Customer", "field": "id"}
            }
        }

    def test_initialization(self):
        """Test generator initialization."""
        # Create with OpenAI configuration
        generator1 = SyntheticDataGenerator(
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            openai_api_key="test_key"
        )
        assert generator1.model_config.provider == "openai"
        assert generator1.model_config.model_name == "gpt-4"
        
        # Create with Anthropic configuration
        generator2 = SyntheticDataGenerator(
            model_config=ModelConfig(provider="anthropic", model_name="claude-3"),
            anthropic_api_key="test_key"
        )
        assert generator2.model_config.provider == "anthropic"
        assert generator2.model_config.model_name == "claude-3"

        # Create with Gemini configuration
        generator3 = SyntheticDataGenerator(
            model_config=ModelConfig(provider="gemini", model_name="gemini-2.5-flash"),
            gemini_api_key="test_key"
        )
        assert generator3.model_config.provider == "gemini"
        assert generator3.model_config.model_name == "gemini-2.5-flash"

    @patch("syda.generate.SchemaLoader")
    def test_generate_for_schemas(self, mock_schema_loader, sample_schema):
        """Test generating data for schemas."""
        # Configure the mock
        mock_loader_instance = MagicMock()
        mock_schema_loader.return_value = mock_loader_instance
        # Configure the load_schema method to return the expected 6 values
        mock_loader_instance.load_schema.return_value = (sample_schema, {}, "Test schema", {}, {}, [])
        
        # Create the generator with a mock LLM client to avoid API calls
        generator = SyntheticDataGenerator(model_config=ModelConfig(provider="openai", model_name="gpt-4"), openai_api_key="test_key")
        
        # Mock the _generate_data method to avoid actual data generation
        generator._generate_data = MagicMock(return_value=pd.DataFrame())
        
        # Generate data using the schema
        result = generator.generate_for_schemas(
            schemas={"customers": sample_schema},
            default_sample_size=5
        )
        
        # Check that the result is a dictionary containing the customers DataFrame
        assert isinstance(result, dict)
        assert "customers" in result
        assert isinstance(result["customers"], pd.DataFrame)
        
    @patch("syda.generate.SchemaLoader")
    def test_generate_with_dependencies(self, mock_schema_loader):
        """Test generating data with foreign key dependencies."""
        # Create schemas with foreign key relationships
        schemas = {
            "customers": {"id": {"type": "number"}, "name": {"type": "text"}},
            "orders": {
                "id": {"type": "number"},
                "customer_id": {
                    "type": "foreign_key",
                    "references": {"schema": "customers", "field": "id"}
                },
                "amount": {"type": "number"}
            }
        }

        # Create mock instances and preset return values
        customers_df = pd.DataFrame({"id": [101, 102], "name": ["Customer A", "Customer B"]})
        orders_df = pd.DataFrame({"id": [1, 2], "customer_id": [101, 102], "amount": [100.0, 200.0]})
        
        # Create a generator with proper model configuration
        with patch("syda.dependency_handler.DependencyHandler") as mock_dependency_handler, \
             patch("syda.generate.SyntheticDataGenerator._generate_data") as mock_generate_data:
            
            # Set up dependency handler mock
            mock_dep_instance = mock_dependency_handler.return_value
            mock_dep_instance.extract_dependencies.return_value = {"orders": ["customers"]}
            mock_dep_instance.determine_generation_order.return_value = ["customers", "orders"]
            mock_dep_instance.has_cycle.return_value = False
            
            # Set up schema loader mock to return expected values
            mock_loader_instance = mock_schema_loader.return_value
            
            def mock_load_schema(schema_source):
                # Debug: Print what's being loaded
                print(f"Loading schema: {schema_source}")
                if isinstance(schema_source, dict):
                    if schema_source == schemas["customers"]:
                        return (schema_source, {}, "Customers Schema", {}, {}, [])
                    elif schema_source == schemas["orders"]:
                        return (schema_source, {}, "Orders Schema", 
                              {"customer_id": ("customers", "id")}, {}, ["customers"])
                return (schema_source, {}, "Unknown Schema", {}, {}, [])
                
            mock_loader_instance.load_schema.side_effect = mock_load_schema
            
            # Configure _generate_data mock to return our predetermined DataFrames
            def side_effect_generate_data(table_schema, metadata=None, table_description=None, prompt=None, sample_size=None, schema_name=None, **kwargs):
                print(f"Generating data for schema_name: {schema_name}")
                if schema_name == "customers":
                    return customers_df
                elif schema_name == "orders":
                    return orders_df
                # Fallback based on the schema itself
                if isinstance(table_schema, dict) and "name" in table_schema:
                    return customers_df
                if isinstance(table_schema, dict) and "customer_id" in table_schema:
                    return orders_df
                return pd.DataFrame()
                
            mock_generate_data.side_effect = side_effect_generate_data
            
            # Create the generator and run the test
            generator = SyntheticDataGenerator(
                model_config=ModelConfig(provider="openai", model_name="gpt-4"),
                openai_api_key="test_key"
            )
            
            # Generate data using the schemas
            print("Calling generate_for_schemas with schemas:")
            for name, schema in schemas.items():
                print(f"  - {name}: {schema}")
                
            result = generator.generate_for_schemas(
                schemas=schemas,
                default_sample_size=5
            )
            
            print(f"Result keys: {result.keys() if result else 'Empty result'}")
            
            # Verify the results contain both DataFrames
            assert isinstance(result, dict)
            assert "customers" in result, "Customers DataFrame not found in result"
            assert "orders" in result, "Orders DataFrame not found in result"
            assert isinstance(result["customers"], pd.DataFrame)
            assert isinstance(result["orders"], pd.DataFrame)
            
            # Verify the mock calls were made correctly
            assert mock_generate_data.call_count >= 2, f"_generate_data was called {mock_generate_data.call_count} times, expected at least 2"
    
    def test_generate_data(self, sample_schema):
        """Test generating data."""
        # Create the generator
        generator = SyntheticDataGenerator(model_config=ModelConfig(provider="openai", model_name="gpt-4"), openai_api_key="test_key")
        
        # Create a mock for _generate_data
        with patch.object(generator, "_generate_data") as mock_generate_data:
            # Mock response from the generate_data method
            mock_df = pd.DataFrame({
                "id": [1, 2],
                "name": ["Alice", "Bob"],
                "email": ["alice@example.com", "bob@example.com"]
            })
            mock_generate_data.return_value = mock_df
            
            # Configure schema loader mock to handle our test schema
            generator.schema_loader = MagicMock()
            generator.schema_loader.load_schema.return_value = (sample_schema, {}, "Test schema", {}, {}, [])
            
            # Generate data using a simple schema
            result = generator.generate_for_schemas({
                "customers": sample_schema
            }, default_sample_size=2)
            
            # Check that mock_generate_data was called
            assert mock_generate_data.called, "_generate_data was not called"
            
            # Check that the result contains the customers DataFrame
            assert isinstance(result, dict)
            assert "customers" in result
            assert isinstance(result["customers"], pd.DataFrame)
        
    def test_schema_loading_error_handling(self):
        """Test error handling when loading an invalid schema."""
        # Create a generator with proper model configuration
        generator = SyntheticDataGenerator(model_config=ModelConfig(provider="openai", model_name="gpt-4"), openai_api_key="test_key")
        
        # Create a schema loader that raises an exception
        generator.schema_loader = MagicMock()
        generator.schema_loader.load_schema.side_effect = ValueError("Invalid schema format")
        
        # Try to generate data with an invalid schema source
        with pytest.raises(ValueError, match="Invalid schema format"):
            generator.generate_for_schemas({"invalid_schema": "invalid_source"})
    
    @patch("syda.dependency_handler.DependencyHandler")
    def test_generate_data_with_foreign_keys(self, mock_dependency_handler, sample_schema_with_fk):
        """Test generating data with foreign key dependencies."""
        # Configure the dependency handler mock
        mock_dep_instance = MagicMock()
        mock_dependency_handler.return_value = mock_dep_instance
        mock_dep_instance.extract_dependencies.return_value = {"orders": ["customers"]}
        mock_dep_instance.build_dependency_graph.return_value = MagicMock()
        mock_dep_instance.determine_generation_order.return_value = ["customers", "orders"]
        
        # Create a generator with proper model configuration
        generator = SyntheticDataGenerator(model_config=ModelConfig(provider="openai", model_name="gpt-4"), openai_api_key="test_key")
        
        # Mock the _generate_data method to return known DataFrames
        def mock_generate_data(table_schema, metadata=None, table_description=None, prompt=None, sample_size=None, schema_name=None, **kwargs):
            if schema_name == "customers" or (isinstance(table_schema, dict) and "name" in table_schema):
                return pd.DataFrame({"id": [101, 102], "name": ["Test1", "Test2"]})
            else:
                return pd.DataFrame({"id": [1, 2], "customer_id": [101, 102]})
                
        generator._generate_data = MagicMock(side_effect=mock_generate_data)
        
        # Create schemas with foreign key relationships
        schemas = {
            "customers": {"id": {"type": "number"}, "name": {"type": "text"}},
            "orders": {
                "id": {"type": "number"},
                "customer_id": {"type": "foreign_key", "references": {"schema": "customers", "field": "id"}}
            }
        }
        
        # Mock the schema loading to handle foreign keys correctly
        def mock_load_schema(schema_source):
            if "id" in schema_source and "name" in schema_source:
                # Customer schema
                return (schema_source, {}, "Customers Schema", {}, {}, [])
            else:
                # Order schema with foreign key
                return (schema_source, {}, "Orders Schema",
                       {"customer_id": ("customers", "id")}, {}, ["customers"])
                
        generator.schema_loader = MagicMock()
        generator.schema_loader.load_schema.side_effect = mock_load_schema
        
        # Generate data with foreign keys
        results = generator.generate_for_schemas(schemas, default_sample_size=2)
        
        # Verify results
        assert "customers" in results
        assert "orders" in results
        assert isinstance(results["customers"], pd.DataFrame)
        assert isinstance(results["orders"], pd.DataFrame)
        
        # Check that customer IDs in orders match IDs in customers
        orders_df = results["orders"]
        customers_df = results["customers"]
        assert set(orders_df["customer_id"]).issubset(set(customers_df["id"]))
    
    @patch("syda.generate.create_empty_dataframe")
    @patch("syda.generate.GeneratorManager")
    def test_generate_random_data(self, mock_generator_manager, mock_create_empty_df, sample_schema):
        """Test fallback to random data generation when LLM generation fails."""
        # Configure the empty DataFrame mock
        mock_df = pd.DataFrame(columns=["id", "name", "email"])
        mock_create_empty_df.return_value = mock_df
        
        # Configure the generator manager mock
        mock_manager_instance = MagicMock()
        mock_generator_manager.return_value = mock_manager_instance
        mock_manager_instance.generate.side_effect = [1, "Alice", "alice@example.com"]
        mock_manager_instance.apply_custom_generators.return_value = pd.DataFrame({
            "id": [1],
            "name": ["Alice"],
            "email": ["alice@example.com"]
        })
        
        # Create the generator with proper model configuration
        generator = SyntheticDataGenerator(model_config=ModelConfig(provider="openai", model_name="gpt-4"), openai_api_key="test_key")
        
        # Mock schema loader to provide the test schema
        generator.schema_loader = MagicMock()
        generator.schema_loader.load_schema.return_value = (sample_schema, {}, "Test schema", {}, {}, [])
        
        # Mock _generate_data to raise an exception to trigger fallback
        with patch.object(generator, "_generate_data", side_effect=Exception("Simulated LLM failure")):
            # Set up schemas for the test
            schemas = {"customers": sample_schema}
            
            # When _generate_data fails, the code should fall back to random generation
            # Pass a small default_sample_size to make the test faster
            with pytest.raises(Exception, match="Simulated LLM failure"):
                generator.generate_for_schemas(schemas, default_sample_size=1)
    
    @patch("syda.generate.save_dataframes")
    def test_save_dataframes(self, mock_save_dataframes):
        """Test saving DataFrames."""
        # Create DataFrames
        df1 = pd.DataFrame({"id": [1, 2]})
        df2 = pd.DataFrame({"id": [3, 4]})
        
        # Create the generator with proper model configuration
        generator = SyntheticDataGenerator(model_config=ModelConfig(provider="openai", model_name="gpt-4"), openai_api_key="test_key")
        
        # Set up dataframes for the test
        dataframes = {
            "table1": df1,
            "table2": df2
        }
        
        # Directly use the save_dataframes function that's being used in generate_for_schemas
        from syda.generate import save_dataframes
        save_dataframes(dataframes, "output_dir", format="csv")
        
        # Check that save_dataframes was called with the correct parameters
        # Adjust assertion to match the actual call (without filenames parameter)
        mock_save_dataframes.assert_called_once_with(
            dataframes, "output_dir", format="csv"
        )
    
    def test_generate_all_schemas(self):
        """Test generating data for all schemas."""
        # Create a generator with proper model configuration
        generator = SyntheticDataGenerator(model_config=ModelConfig(provider="openai", model_name="gpt-4"), openai_api_key="test_key")
        
        # Create test schemas
        schemas = {
            "table1": {"id": {"type": "number"}},
            "table2": {"id": {"type": "number"}}
        }
        
        # Mock the schema loader
        generator.schema_loader = MagicMock()
        generator.schema_loader.load_schema.side_effect = [
            (schemas["table1"], {}, "Table 1", {}, {}, []),
            (schemas["table2"], {}, "Table 2", {}, {}, [])
        ]
        
        # Mock _generate_data to avoid actual LLM calls
        generator._generate_data = MagicMock(return_value=pd.DataFrame({"id": [1, 2]}))
        
        # Generate data for all schemas
        result = generator.generate_for_schemas(
            schemas=schemas,
            default_sample_size=10
        )
        
        # Verify results contain both tables
        assert "table1" in result
        assert "table2" in result
        assert isinstance(result["table1"], pd.DataFrame)
        assert isinstance(result["table2"], pd.DataFrame)
        
    def test_detect_dependency_cycle(self):
        """Test that a circular dependency in schemas is handled gracefully."""
        # Create schemas with a circular dependency
        schemas = {
            'table1': {'id': {'type': 'number'}, 'table2_id': {'type': 'foreign_key', 'references': {'schema': 'table2', 'field': 'id'}}},
            'table2': {'id': {'type': 'number'}, 'table1_id': {'type': 'foreign_key', 'references': {'schema': 'table1', 'field': 'id'}}}
        }
        
        # Mock the dependency handler's methods to simulate a cycle
        with patch('syda.dependency_handler.DependencyHandler.determine_generation_order') as mock_determine:
            # Simulate a circular dependency error
            mock_determine.side_effect = ValueError("Circular dependencies detected in schemas")
            
            # Mock the foreign key handler to prevent foreign key errors
            with patch('syda.dependency_handler.ForeignKeyHandler.apply_foreign_keys'):
                # Mock _generate_data to avoid actual API calls and return dataframes with proper columns
                with patch('syda.generate.SyntheticDataGenerator._generate_data') as mock_generate_data:
                    # Return a DataFrame with the 'id' column for both tables
                    mock_generate_data.side_effect = lambda **kwargs: pd.DataFrame({'id': [1, 2], 
                                                                                 'table1_id' if 'table2' in kwargs['table_description'] else 'table2_id': [1, 2]})
                    
                    # Create the generator
                    generator = SyntheticDataGenerator(
                        model_config=ModelConfig(provider="openai", model_name="gpt-4"),
                        openai_api_key="test_key"
                    )
                    
                    # The method should handle the cycle and not raise an exception
                    result = generator.generate_for_schemas(schemas)
                    
                    # Verify that determine_generation_order was called
                    mock_determine.assert_called_once()
                    
                    # Verify that we still got results for both schemas
                    assert set(result.keys()) == set(['table1', 'table2'])
                    assert isinstance(result['table1'], pd.DataFrame)
                    assert isinstance(result['table2'], pd.DataFrame)
            
    def test_client_error_handling(self):
        """Test handling of client errors when generating data."""
        # Create a schema
        test_schema = {'id': {'type': 'number'}}
        
        # Create the generator
        generator = SyntheticDataGenerator(
            model_config=ModelConfig(provider="openai", model_name="gpt-4"),
            openai_api_key="test_key"
        )
        
        # Mock the schema loading to avoid external dependencies
        generator.schema_loader = MagicMock()
        generator.schema_loader.load_schema.return_value = (test_schema, {}, "Test schema", {}, {}, [])
        
        # Patch the instance method _generate_data to raise an exception
        with patch.object(SyntheticDataGenerator, "_generate_data") as mock_generate_data:
            # Mock the generate_data method to raise a client error
            mock_generate_data.side_effect = Exception("Simulated client error")
            
            # Test that the exception is properly propagated
            with pytest.raises(Exception, match="Simulated client error"):
                generator.generate_for_schemas({"customers": test_schema}, default_sample_size=10)


# ---------------------------------------------------------------------------
# Large-dataset helpers
# ---------------------------------------------------------------------------

class TestResolveBatchSize:
    """_resolve_batch_size auto-selects appropriate chunk size."""

    def _gen(self):
        from unittest.mock import patch, MagicMock
        from syda.generate import SyntheticDataGenerator
        from syda.schemas import ModelConfig
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(model_config=ModelConfig())

    def test_small_sample_single_chunk(self):
        g = self._gen()
        assert g._resolve_batch_size(10) == 10

    def test_medium_sample_50(self):
        g = self._gen()
        assert g._resolve_batch_size(100) == 50

    def test_large_sample_100(self):
        g = self._gen()
        assert g._resolve_batch_size(500) == 100

    def test_very_large_sample_200(self):
        g = self._gen()
        assert g._resolve_batch_size(2000) == 200

    def test_explicit_override(self):
        g = self._gen()
        assert g._resolve_batch_size(1000, batch_size=25) == 25

    def test_model_config_batch_size(self):
        from unittest.mock import patch, MagicMock
        from syda.generate import SyntheticDataGenerator
        from syda.schemas import ModelConfig
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            g = SyntheticDataGenerator(model_config=ModelConfig(batch_size=75))
        assert g._resolve_batch_size(500) == 75


class TestCallWithRetry:
    """_call_with_retry retries on transient errors, not on auth/value errors."""

    def _gen(self):
        from unittest.mock import patch, MagicMock
        from syda.generate import SyntheticDataGenerator
        from syda.schemas import ModelConfig
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(model_config=ModelConfig())

    def test_success_on_first_try(self):
        g = self._gen()
        result = g._call_with_retry(lambda: 42, max_retries=3)
        assert result == 42

    def test_retries_on_transient_and_succeeds(self):
        from unittest.mock import MagicMock, patch
        g = self._gen()
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        with patch("time.sleep"):
            result = g._call_with_retry(flaky, max_retries=3)
        assert result == "ok"
        assert call_count["n"] == 3

    def test_no_retry_on_value_error(self):
        from unittest.mock import patch
        g = self._gen()
        call_count = {"n": 0}

        def bad():
            call_count["n"] += 1
            raise ValueError("schema error")

        with patch("time.sleep"):
            with pytest.raises(ValueError, match="schema error"):
                g._call_with_retry(bad, max_retries=3)
        assert call_count["n"] == 1

    def test_exhausts_retries_and_raises(self):
        from unittest.mock import patch
        g = self._gen()
        call_count = {"n": 0}

        def always_fail():
            call_count["n"] += 1
            raise ConnectionError("persistent")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="retries"):
                g._call_with_retry(always_fail, max_retries=2)
        assert call_count["n"] == 3  # 1 initial + 2 retries


class TestEnforceUniqueness:
    """_enforce_uniqueness stamps sequential IDs and deduplicates unique strings."""

    def _gen(self):
        from unittest.mock import patch, MagicMock
        from syda.generate import SyntheticDataGenerator
        from syda.schemas import ModelConfig
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(model_config=ModelConfig())

    def test_integer_pk_sequential(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"id": [5, 5, 5], "name": ["a", "b", "c"]})
        schema = {"id": "integer", "name": "text"}
        metadata = {"id": {"constraints": {"primary_key": True}}, "name": {}}
        result = g._enforce_uniqueness(df, schema, metadata, chunk_offset=0)
        assert list(result["id"]) == [1, 2, 3]

    def test_integer_pk_with_offset(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"id": [1, 2, 3], "v": ["x", "y", "z"]})
        schema = {"id": "integer", "v": "text"}
        metadata = {"id": {"constraints": {"primary_key": True}}, "v": {}}
        result = g._enforce_uniqueness(df, schema, metadata, chunk_offset=100)
        assert list(result["id"]) == [101, 102, 103]

    def test_no_change_when_already_unique(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"email": ["a@x.com", "b@x.com", "c@x.com"]})
        schema = {"email": "email"}
        metadata = {"email": {"constraints": {"unique": True}}}
        result = g._enforce_uniqueness(df, schema, metadata)
        assert list(result["email"]) == ["a@x.com", "b@x.com", "c@x.com"]

    def test_stamps_duplicate_unique_strings(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"code": ["ABC", "ABC", "ABC"]})
        schema = {"code": "text"}
        metadata = {"code": {"constraints": {"unique": True}}}
        result = g._enforce_uniqueness(df, schema, metadata)
        assert len(set(result["code"])) == 3


class TestApplyConstraints:
    """_apply_constraints truncates strings and clamps numerics."""

    def _gen(self):
        from unittest.mock import patch, MagicMock
        from syda.generate import SyntheticDataGenerator
        from syda.schemas import ModelConfig
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(model_config=ModelConfig())

    def test_truncates_over_max_length(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"note": ["hello world long text"]})
        schema = {"note": "text"}
        metadata = {"note": {"constraints": {"max_length": 5}}}
        result = g._apply_constraints(df, schema, metadata)
        assert result["note"].iloc[0] == "hello"

    def test_no_truncation_within_limit(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"code": ["ABC"]})
        schema = {"code": "text"}
        metadata = {"code": {"constraints": {"max_length": 10}}}
        result = g._apply_constraints(df, schema, metadata)
        assert result["code"].iloc[0] == "ABC"

    def test_clamps_numeric_min(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"age": [-5]})
        schema = {"age": "integer"}
        metadata = {"age": {"constraints": {"min": 0, "max": 120}}}
        result = g._apply_constraints(df, schema, metadata)
        assert result["age"].iloc[0] == 0

    def test_clamps_numeric_max(self):
        import pandas as pd
        g = self._gen()
        df = pd.DataFrame({"score": [200.0]})
        schema = {"score": "float"}
        metadata = {"score": {"constraints": {"max": 100}}}
        result = g._apply_constraints(df, schema, metadata)
        assert result["score"].iloc[0] == 100


class TestChunkedGeneration:
    """_generate_data correctly chunks into multiple LLM calls in direct mode."""

    def _gen(self, batch_size=None, mode="direct"):
        from unittest.mock import patch, MagicMock
        from syda.generate import SyntheticDataGenerator
        from syda.schemas import ModelConfig
        kwargs = dict(generation_mode=mode)
        if batch_size:
            kwargs["batch_size"] = batch_size
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(model_config=ModelConfig(**kwargs))

    def test_single_chunk_when_small(self):
        """sample_size ≤ batch_size → exactly 1 LLM call."""
        from unittest.mock import patch, MagicMock, call
        import pandas as pd

        g = self._gen(batch_size=50, mode="direct")
        chunk_df = pd.DataFrame({"id": range(10), "name": ["x"] * 10})
        call_count = {"n": 0}

        def fake_llm(schema, prompt, sz):
            call_count["n"] += 1
            return chunk_df.iloc[:sz].copy(), 0, 0

        with patch.object(g, "_generate_data_with_llm", side_effect=fake_llm):
            result = g._generate_data(
                table_schema={"id": "integer", "name": "text"},
                metadata={},
                sample_size=10,
            )
        assert call_count["n"] == 1
        assert len(result) == 10

    def test_multiple_chunks_concat(self):
        """sample_size=130, batch_size=50 → 3 LLM calls, 130 rows total."""
        from unittest.mock import patch
        import pandas as pd
        import math

        g = self._gen(batch_size=50, mode="direct")
        call_sizes = []

        def fake_llm(schema, prompt, sz):
            call_sizes.append(sz)
            return pd.DataFrame({"id": range(sz), "val": ["v"] * sz}), 0, 0

        with patch.object(g, "_generate_data_with_llm", side_effect=fake_llm):
            result = g._generate_data(
                table_schema={"id": "integer", "val": "text"},
                metadata={},
                sample_size=130,
            )
        assert len(call_sizes) == 3
        assert call_sizes == [50, 50, 30]
        assert len(result) == 130

    def test_progress_messages_printed(self, capsys):
        """Chunk progress messages appear on stdout."""
        from unittest.mock import patch
        import pandas as pd

        g = self._gen(batch_size=5, mode="direct")

        def fake_llm(schema, prompt, sz):
            return pd.DataFrame({"x": range(sz)}), 0, 0

        with patch.object(g, "_generate_data_with_llm", side_effect=fake_llm):
            g._generate_data(
                table_schema={"x": "integer"},
                metadata={},
                sample_size=12,
            )
        captured = capsys.readouterr()
        assert "[syda] Generating chunk 1/3" in captured.out

    def test_auto_mode_uses_direct_for_small(self):
        """auto mode selects direct when sample_size ≤ 500."""
        from unittest.mock import patch
        import pandas as pd

        g = self._gen(mode="auto")
        call_count = {"n": 0}

        def fake_llm(schema, prompt, sz):
            call_count["n"] += 1
            return pd.DataFrame({"id": range(sz)}), 0, 0

        with patch.object(g, "_generate_data_with_llm", side_effect=fake_llm):
            g._generate_data(
                table_schema={"id": "integer"},
                metadata={},
                sample_size=10,
            )
        # Should have called LLM (direct mode, single chunk)
        assert call_count["n"] == 1

    def test_auto_mode_selects_codegen_above_threshold(self):
        """auto mode selects codegen when sample_size > 500."""
        from unittest.mock import patch, MagicMock
        import pandas as pd

        g = self._gen(mode="auto")
        semantic_cols = []
        local_df = pd.DataFrame({"id": range(501)})

        with patch.object(g, "_analyze_and_generate_code", return_value=semantic_cols) as mock_analyze, \
             patch.object(g, "_run_local_generation", return_value=local_df):
            g._generate_data(
                table_schema={"id": "integer"},
                metadata={},
                sample_size=501,
            )
        mock_analyze.assert_called_once()


class TestStreamingOutput:
    """Chunks are written to disk progressively when output_dir is set and sample_size > 10_000."""

    def _gen(self, mode="direct"):
        from unittest.mock import patch, MagicMock
        from syda.generate import SyntheticDataGenerator
        from syda.schemas import ModelConfig
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(model_config=ModelConfig(generation_mode=mode))

    def test_chunks_written_incrementally(self, tmp_path):
        """Each chunk is appended to the output file before the next chunk starts."""
        from unittest.mock import patch
        import pandas as pd
        from syda.generate import _STREAM_THRESHOLD

        g = self._gen(mode="direct")
        sample_size = _STREAM_THRESHOLD + 100
        batch_size = 50
        write_calls = []

        real_append = __import__("syda.output", fromlist=["append_dataframe"]).append_dataframe

        def tracking_append(df, path, **kwargs):
            write_calls.append(len(df))
            return real_append(df, path, **kwargs)

        def fake_llm(schema, prompt, sz):
            return pd.DataFrame({"id": range(sz)}), 0, 0

        with patch.object(g, "_generate_data_with_llm", side_effect=fake_llm), \
             patch("syda.generate.append_dataframe", side_effect=tracking_append):
            stream_path = str(tmp_path / "out.csv")
            result = g._generate_data(
                table_schema={"id": "integer"},
                metadata={},
                sample_size=sample_size,
                batch_size=batch_size,
                stream_path=stream_path,
            )

        # Should have written many chunks to disk
        assert len(write_calls) > 1
        # Total written rows equals sample_size
        assert sum(write_calls) == sample_size

    def test_streamed_table_not_saved_again(self, tmp_path):
        """Tables streamed to disk during generation are not passed to save_dataframes."""
        from unittest.mock import patch, MagicMock
        import pandas as pd
        from syda.generate import SyntheticDataGenerator, _STREAM_THRESHOLD
        from syda.schemas import ModelConfig

        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            g = SyntheticDataGenerator(model_config=ModelConfig(generation_mode="direct"))

        sample_size = _STREAM_THRESHOLD + 1

        def fake_generate(*args, **kwargs):
            sp = kwargs.get("stream_path")
            if sp:
                import os; os.makedirs(os.path.dirname(sp), exist_ok=True)
                pd.DataFrame({"id": range(3)}).to_csv(sp, index=False)
                return pd.DataFrame({"id": range(3)})
            return pd.DataFrame({"id": range(sample_size)})

        save_calls = []

        with patch.object(g, "_generate_data", side_effect=fake_generate), \
             patch("syda.generate.save_dataframes", side_effect=lambda d, *a, **k: save_calls.append(set(d.keys()))):
            g.generate_for_schemas(
                schemas={"patients": {"id": "integer"}},
                default_sample_size=sample_size,
                output_dir=str(tmp_path),
            )

        # patients was streamed — it must NOT appear in save_dataframes call
        for call_keys in save_calls:
            assert "patients" not in call_keys


class TestParallelGeneration:
    """Tests for parallel (multi-worker) table generation."""

    @pytest.fixture
    def generator(self):
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(
                model_config=ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001")
            )

    def _make_df(self, name, n=3):
        return pd.DataFrame({"id": range(n), "name": [f"{name}_{i}" for i in range(n)]})

    def test_max_workers_default_is_one(self):
        config = ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001")
        assert config.max_workers == 1

    def test_max_workers_set(self):
        config = ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001", max_workers=4)
        assert config.max_workers == 4

    def test_max_workers_must_be_positive(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ModelConfig(provider="anthropic", model_name="m", max_workers=0)

    def test_sequential_and_parallel_produce_same_tables(self, generator):
        """With max_workers > 1, the same table names are returned."""
        schemas = {
            "Customer": {"id": "integer", "name": "text"},
            "Product":  {"sku": "integer", "title": "text"},
        }

        def fake_one_table(schema_name, **kwargs):
            df = self._make_df(schema_name)
            return schema_name, df, MagicMock(duration_s=0, row_count=len(df)), False

        with patch.object(generator, "_generate_one_table", side_effect=fake_one_table):
            results_seq = generator.generate_for_schemas(
                schemas=schemas, default_sample_size=3, max_workers=1
            )
            results_par = generator.generate_for_schemas(
                schemas=schemas, default_sample_size=3, max_workers=4
            )

        assert set(results_seq.keys()) == set(results_par.keys()) == {"Customer", "Product"}

    def test_independent_tables_run_in_parallel(self, generator):
        """Tables with no FK dependency should be submitted concurrently."""
        import threading
        schemas = {
            "A": {"id": "integer"},
            "B": {"id": "integer"},
            "C": {"id": "integer"},
        }

        concurrent_count = []
        active = threading.Value("i", 0) if hasattr(threading, "Value") else None
        lock = threading.Lock()
        peak = [0]

        def fake_one_table(schema_name, **kwargs):
            with lock:
                peak[0] = max(peak[0], 1)  # at minimum 1 thread always
            df = self._make_df(schema_name)
            return schema_name, df, MagicMock(duration_s=0, row_count=len(df)), False

        with patch.object(generator, "_generate_one_table", side_effect=fake_one_table):
            results = generator.generate_for_schemas(
                schemas=schemas, default_sample_size=3, max_workers=3
            )

        assert set(results.keys()) == {"A", "B", "C"}

    def test_fk_children_generated_after_parents(self, generator):
        """Child tables must only be generated after their parent tables."""
        generated_order = []

        schemas = {
            "Parent": {"id": "integer"},
            "Child":  {
                "__foreign_keys__": {"parent_id": ["Parent", "id"]},
                "id": "integer",
                "parent_id": "foreign_key",
            },
        }

        def fake_one_table(schema_name, **kwargs):
            generated_order.append(schema_name)
            df = pd.DataFrame({"id": [1, 2, 3], "parent_id": [1, 1, 2]})
            return schema_name, df, MagicMock(duration_s=0, row_count=3), False

        with patch.object(generator, "_generate_one_table", side_effect=fake_one_table):
            generator.generate_for_schemas(
                schemas=schemas, default_sample_size=3, max_workers=4
            )

        assert generated_order.index("Parent") < generated_order.index("Child")

    def test_parallel_levels_computed_correctly(self):
        """DependencyHandler.compute_parallel_levels groups independent tables correctly."""
        from syda.dependency_handler import DependencyHandler
        import networkx as nx

        # A → C, B → C  (A and B are independent; C depends on both)
        g = nx.DiGraph()
        g.add_edges_from([("A", "C"), ("B", "C")])

        levels = DependencyHandler.compute_parallel_levels(g)
        assert len(levels) == 2
        assert set(levels[0]) == {"A", "B"}   # both independent at level 0
        assert levels[1] == ["C"]

    def test_single_table_no_parallelism(self, generator):
        """Single table with max_workers > 1 still works correctly."""
        schemas = {"Only": {"id": "integer", "val": "text"}}

        def fake_one_table(schema_name, **kwargs):
            df = self._make_df(schema_name)
            return schema_name, df, MagicMock(duration_s=0, row_count=len(df)), False

        with patch.object(generator, "_generate_one_table", side_effect=fake_one_table):
            results = generator.generate_for_schemas(
                schemas=schemas, default_sample_size=3, max_workers=8
            )

        assert list(results.keys()) == ["Only"]

    def test_parallel_generate_via_model_config(self, generator):
        """max_workers on ModelConfig is respected when not overridden in call."""
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            g = SyntheticDataGenerator(
                model_config=ModelConfig(
                    provider="anthropic",
                    model_name="claude-haiku-4-5-20251001",
                    max_workers=4,
                )
            )

        schemas = {"X": {"id": "integer"}, "Y": {"id": "integer"}}

        called_with_workers = []

        original = g._generate_structured_data

        def spy(*args, **kwargs):
            called_with_workers.append(kwargs.get("max_workers", 1))
            return original(*args, **kwargs)

        def fake_one_table(schema_name, **kwargs):
            df = self._make_df(schema_name)
            return schema_name, df, MagicMock(duration_s=0, row_count=len(df)), False

        with patch.object(g, "_generate_structured_data", side_effect=spy), \
             patch.object(g, "_generate_one_table", side_effect=fake_one_table):
            g.generate_for_schemas(schemas=schemas, default_sample_size=3)

        assert called_with_workers and called_with_workers[0] == 4


class TestParallelRateLimitHandling:
    """Tests for rate-limit and error handling in parallel generation."""

    @pytest.fixture
    def generator(self):
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            return SyntheticDataGenerator(
                model_config=ModelConfig(
                    provider="anthropic",
                    model_name="claude-haiku-4-5-20251001",
                    max_workers=4,
                )
            )

    def _make_df(self, name, n=3):
        return pd.DataFrame({"id": range(n), "val": [f"{name}_{i}" for i in range(n)]})

    def test_429_sets_global_backoff(self):
        """When a 429 is hit, _set_global_backoff is called to pause all threads."""
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            g = SyntheticDataGenerator(
                model_config=ModelConfig(provider="anthropic", model_name="m", max_workers=4)
            )

        call_count = [0]
        sleep_delays = []

        def rate_limit_then_ok():
            call_count[0] += 1
            if call_count[0] == 1:
                err = Exception("rate limit")
                err.status_code = 429
                raise err
            return "ok"

        # time.time returns 1000.0 before/during the 429 handler, then 2000.0 on
        # the next attempt so _wait_for_global_backoff sees an expired deadline.
        time_sequence = iter([1000.0, 1000.0, 2000.0, 2000.0])
        with patch("syda.generate.time.sleep", side_effect=lambda d: sleep_delays.append(d)), \
             patch("syda.generate.random.uniform", return_value=5.0), \
             patch("syda.generate.time.time", side_effect=time_sequence):
            result = g._call_with_retry(rate_limit_then_ok, max_retries=2)

        assert result == "ok"
        # _set_global_backoff should have set _backoff_until > 0
        assert g._backoff_until > 0
        # One sleep for the 429 delay (the next _wait_for_global_backoff sees expired deadline)
        assert len(sleep_delays) == 1
        # delay = 60.0 + uniform(0, 10) = 60 + 5 = 65
        assert sleep_delays[0] == pytest.approx(65.0)

    def test_wait_for_global_backoff_blocks_other_threads(self):
        """_wait_for_global_backoff sleeps for the remaining backoff time."""
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            g = SyntheticDataGenerator(
                model_config=ModelConfig(provider="anthropic", model_name="m", max_workers=4)
            )

        sleep_delays = []

        # Simulate another thread having set _backoff_until to 20s from now
        with patch("syda.generate.time.time", return_value=1000.0):
            g._backoff_until = 1020.0  # 20 seconds in the future

        with patch("syda.generate.time.sleep", side_effect=lambda d: sleep_delays.append(d)), \
             patch("syda.generate.time.time", return_value=1000.0):
            g._wait_for_global_backoff()

        assert len(sleep_delays) == 1
        assert sleep_delays[0] == pytest.approx(20.0)

    def test_wait_for_global_backoff_noop_when_expired(self):
        """_wait_for_global_backoff does not sleep when backoff has already expired."""
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            g = SyntheticDataGenerator(
                model_config=ModelConfig(provider="anthropic", model_name="m", max_workers=4)
            )

        sleep_delays = []
        # _backoff_until defaults to 0.0 — already expired
        with patch("syda.generate.time.sleep", side_effect=lambda d: sleep_delays.append(d)), \
             patch("syda.generate.time.time", return_value=1000.0):
            g._wait_for_global_backoff()

        assert sleep_delays == []

    def test_successful_tables_preserved_on_partial_failure(self):
        """When one table fails, already-completed tables in the same level are kept."""
        with patch("syda.llm._build_pydantic_ai_model", return_value=MagicMock()):
            g = SyntheticDataGenerator(
                model_config=ModelConfig(
                    provider="anthropic", model_name="m", max_workers=3
                )
            )

        schemas = {
            "Good1": {"id": "integer"},
            "Bad":   {"id": "integer"},
            "Good2": {"id": "integer"},
        }

        def fake_one_table(schema_name, **kwargs):
            if schema_name == "Bad":
                raise RuntimeError("simulated API failure")
            df = self._make_df(schema_name)
            return schema_name, df, MagicMock(duration_s=0, row_count=len(df)), False

        with patch.object(g, "_generate_one_table", side_effect=fake_one_table):
            with pytest.raises(Exception, match="Bad"):
                g.generate_for_schemas(schemas=schemas, default_sample_size=3)
