"""
Structured synthetic data generation using LLMs with SQLAlchemy integration.
"""

import pandas as pd
import json
import os
import random
import pkgutil
import importlib
import inspect
from pathlib import Path
import networkx as nx
from typing import Dict, List, Optional, Callable, Union, Type, Any, Tuple
from pydantic import create_model, TypeAdapter
from .schemas import ModelConfig
from .llm import create_llm_client, LLMClient
from .output import save_dataframe, save_dataframes
from .utils import (
    sqlalchemy_model_to_schema,
    create_empty_dataframe,
    generate_random_value,
    get_schema_prompt,
    parse_dataframe_output
)

try:
    from sqlalchemy.orm import DeclarativeMeta
    from sqlalchemy import inspect
except ImportError:
    DeclarativeMeta = None


class SyntheticDataGenerator:
    """Generator for synthetic data using LLMs."""
    
    def __init__(self, model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None):
        """
        Initialize the synthetic data generator with the specified model configuration.
        
        Args:
            model_config: Configuration for the AI model to use, either as a ModelConfig object 
                         or a dictionary of parameters. If None, default settings will be used.
            openai_api_key: Optional API key for OpenAI. If not provided, will use OPENAI_API_KEY 
                           environment variable.
            anthropic_api_key: Optional API key for Anthropic. If not provided, will use 
                              ANTHROPIC_API_KEY environment variable.
        """
        # Initialize the LLM client using our new module
        self.llm_client = create_llm_client(
            model_config=model_config,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key
        )
        
        # Store the model configuration for easy access
        self.model_config = self.llm_client.model_config
        
        # Access the instructor client directly
        self.client = self.llm_client.client
        
        # Registry for custom generators by type: type_name -> fn(row: pd.Series, col_name: str) -> value
        self.type_generators: Dict[str, Callable[[pd.Series, str], any]] = {}
        
        # Registry for custom generators by column name: col_name -> fn(row: pd.Series, col_name: str) -> value
        self.column_generators: Dict[str, Callable[[pd.Series, str], any]] = {}

    def register_generator(self, type_name: str, func: Callable[[pd.Series, str], any], column_name: Optional[str] = None):
        """
        Register a custom generator for a specific data type or column name.
        
        Args:
            type_name: The data type this generator handles (e.g., 'number', 'text', 'foreign_key')
            func: Function that takes (row: pd.Series, col_name: str) and returns a generated value
            column_name: If specified, this generator only applies to the named column
                        rather than all columns of the specified type
        """
        if column_name:
            # Register a column-specific generator
            self.column_generators[column_name] = func
        else:
            # Register a type-based generator
            self.type_generators[type_name.lower()] = func

    def _build_dependency_graph(self, nodes, dependencies):
        """
        Generic method to build a directed graph of dependencies.
        
        Args:
            nodes: List of node names to add to the graph
            dependencies: Dict mapping node names to their dependencies
            
        Returns:
            NetworkX DiGraph representing dependencies between nodes
        """
        import networkx as nx
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all nodes to the graph
        for node in nodes:
            G.add_node(node)
        
        # Add edges based on dependencies
        for dependent, dependencies_list in dependencies.items():
            for dependency in dependencies_list:
                if dependency != dependent:  # Avoid self-references
                    # Add an edge from dependency to dependent
                    # (dependency must be populated before dependent)
                    G.add_edge(dependency, dependent)
        
        return G

    def _sqlalchemy_models_to_schemas(self, sqlalchemy_models):
        """
        Convert a list of SQLAlchemy models to the schema format expected by generate_for_schemas.
        
        Args:
            sqlalchemy_models: List of SQLAlchemy model classes
            
        Returns:
            Dictionary mapping schema names to schema definitions
        """
        from syda.utils import sqlalchemy_model_to_schema
        
        schemas = {}
        
        for model_class in sqlalchemy_models:
            model_name = model_class.__name__
            
            # Convert SQLAlchemy model to schema format
            # Get table name, schema dict, and metadata dict
            table_name, schema_dict, metadata_dict = sqlalchemy_model_to_schema(model_class)
            
            # Use table_name as the key in our schemas dictionary (not model_name)
            schema_name = table_name
            description = metadata_dict.get('__description__', f"{schema_name} data")
            
            # Combine schema with metadata in format expected by generate_for_schemas
            combined_schema = schema_dict.copy()
            
            # Add regular fields
            for key, value in schema_dict.items():
                if not key.startswith('__') and not isinstance(value, str):
                    combined_schema[key] = value
            
            # Add special fields with double underscores
            for key, value in schema_dict.items():
                if key.startswith('__'):
                    combined_schema[key] = value

            # Handle case where schema_dict[key] is a string (type)
            # such as schema_dict['field_name'] = 'text'
            # This is the normal format in our schema dicts
            # Add metadata attributes from metadata_dict
            for key, value in metadata_dict.items():
                if key.startswith('__') and key.endswith('__'):
                    combined_schema[key] = value
            
            # Add description if available
            if description:
                combined_schema['__description__'] = description
            
            # Special handling for template dependencies
            if hasattr(model_class, '__depends_on__'):
                combined_schema['__depends_on__'] = getattr(model_class, '__depends_on__')
            
            # Special handling for template classes
            if '__template_source__' in metadata_dict:
                # Copy template source to actual value
                combined_schema['template_source'] = metadata_dict['__template_source__']
                combined_schema['input_file_type'] = metadata_dict.get('__input_file_type__', 'html')
                combined_schema['output_file_type'] = metadata_dict.get('__output_file_type__', 'pdf')
            
            # Add to schemas dictionary
            schemas[schema_name] = combined_schema
            
        return schemas
    
    def _handle_missing_columns(self, df, model_name, schema, foreign_key_info=None, custom_generators=None):
        """
        Add any missing columns to the DataFrame with appropriate values.
        
        Args:
            df: DataFrame to modify
            model_name: Name of the model
            schema: Schema definition
            foreign_key_info: Optional dict with foreign key information
            custom_generators: Optional dict with custom generators
            
        Returns:
            pd.DataFrame: DataFrame with all required columns
        """
        for col_name in schema.keys():
            if col_name not in df.columns:
                print(f"⚠️ Adding missing column '{col_name}' to {model_name}")
                # Check if this is a foreign key column first
                if foreign_key_info and col_name in foreign_key_info and foreign_key_info[col_name]['values']:
                    fk_info = foreign_key_info[col_name]
                    print(f"  Using foreign key values from {fk_info['target_model']}.{fk_info['target_col']} for {col_name}")
                    df[col_name] = pd.Series([random.choice(fk_info['values']) for _ in range(len(df))])
                else:
                    # Register custom generator if available
                    if custom_generators and model_name in custom_generators and col_name in custom_generators[model_name]:
                        print(f"  Using custom generator for {col_name}")
                        self.register_generator(schema[col_name], custom_generators[model_name][col_name], column_name=col_name)
                    
                    # Generate missing column with appropriate data
                    col_type = schema[col_name]
                    if col_name in self.column_generators:
                        # Use column-specific generator
                        df[col_name] = pd.Series([None for _ in range(len(df))])
                    elif col_type.lower() in self.type_generators:
                        # Use type-based generator
                        df[col_name] = pd.Series([None for _ in range(len(df))])
                    else:
                        # Generate placeholder values based on column type
                        df[col_name] = pd.Series([generate_random_value(col_type) for _ in range(len(df))])
        return df

    def generate_for_sqlalchemy_models(
        self,
        sqlalchemy_models: Union[List[Type], Type, str],
        prompts: Optional[Dict[str, str]] = None,
        sample_sizes: Optional[Dict[str, int]] = None,
        output_dir: Optional[str] = None,
        default_sample_size: int = 10,
        default_prompt: str = "Generate synthetic data",
        custom_generators: Optional[Dict[str, Dict[str, Callable]]] = None,
        output_format: str = 'csv'
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for multiple relational SQLAlchemy models with automatic 
        dependency resolution based on foreign key relationships.
        
        This function supports both regular SQLAlchemy models and pure template classes
        (without SQLAlchemy tables) by converting them to schema format and then using
        the generate_for_schemas method for data generation.
        
        This function:
        1. Converts SQLAlchemy models to schema format
        2. Calls generate_for_schemas to handle the data generation
        3. This ensures consistent handling of both regular models and template classes
        
        Args:
            sqlalchemy_models: A list of SQLAlchemy model classes, a single SQLAlchemy model class, 
                    or a string pattern to match class names
            prompts: Optional dictionary mapping model names to custom prompts
            sample_sizes: Optional dictionary mapping model names to sample sizes
            output_dir: Optional directory to save files (one per model)
            default_sample_size: Default number of records if not specified in sample_sizes
            default_prompt: Default prompt if not specified in prompts
            custom_generators: Optional dictionary specifying custom generators for SQLAlchemy models and columns.
                              Format: {"ModelName": {"column_name": generator_function}}
            output_format: Format to use when saving files ('csv' or 'json')
            
        Returns:
            Dictionary mapping model names to DataFrames of generated data
        """
        # Initialize default parameters
        if prompts is None:
            prompts = {}
        if sample_sizes is None:
            sample_sizes = {}
        if custom_generators is None:
            custom_generators = {}
            
        # Handle single SQLAlchemy model case
        if not isinstance(sqlalchemy_models, list):
            sqlalchemy_models = [sqlalchemy_models]
        
        # Convert SQLAlchemy models to schema format
        schemas = self._sqlalchemy_models_to_schemas(sqlalchemy_models)
        
        # Call generate_for_schemas with the converted schemas
        return self.generate_for_schemas(
            schemas=schemas,
            prompts=prompts,
            sample_sizes=sample_sizes,
            custom_generators=custom_generators,
            output_dir=output_dir,
            default_sample_size=default_sample_size,
            default_prompt=default_prompt,
            output_format=output_format
        )
        
    def generate_for_schemas(
        self,
        schemas: Dict[str, Union[Dict[str, str], str]],
        prompts: Optional[Dict[str, str]] = None,
        sample_sizes: Optional[Dict[str, int]] = None,
        output_dir: Optional[str] = None,
        default_sample_size: int = 10,
        default_prompt: str = "Generate synthetic data",
        custom_generators: Optional[Dict[str, Dict[str, Callable]]] = None,
        output_format: str = 'csv'
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for multiple related schemas with automatic 
        dependency resolution based on foreign key relationships.
        
        This function supports different schema input formats:
        - Dictionary schemas directly in the code
        - JSON schema files (.json)
        - YAML schema files (.yml, .yaml)
        
        This function:
        1. Loads schema definitions from various sources
        2. Analyzes schema dependencies using foreign keys defined in schemas
        3. Automatically determines the correct order to generate data
        4. Handles foreign key relationships between schemas
        5. Applies custom generators where registered
        
        Foreign key relationships can be defined in three ways:
        
        1. Using the '__foreign_keys__' special section in a schema:
           "__foreign_keys__": {
               "customer_id": ["Customer", "id"]
           }
        
        2. Using field-level references with type and references properties:
           "order_id": {
               "type": "foreign_key",
               "references": {
                   "schema": "Order",
                   "field": "id"
               }
           }
        
        3. Using type-based detection with naming conventions:
           "customer_id": "foreign_key"
           (The system will attempt to infer the relationship based on naming conventions)
        
        
        Args:
            schemas: Dictionary mapping schema names to either:
                    - Schema dictionaries (e.g., {'id': 'number', 'name': 'text'})
                    - File paths to JSON or YAML schema files
            prompts: Optional dictionary mapping schema names to custom prompts
            sample_sizes: Optional dictionary mapping schema names to sample sizes
            output_dir: Optional directory to save files (one per schema)
            default_sample_size: Default number of records if not specified in sample_sizes
            default_prompt: Default prompt if not specified in prompts
            custom_generators: Optional dictionary specifying custom generators for schemas and columns.
                               Custom generator functions can accept the following parameters:
                               - row: The current row being processed
                               - col_name: The name of the column being generated
                               - parent_dfs: Dictionary of previously generated dataframes (schema name as key)
                               
                               Example of a custom generator using parent dataframes:
                               
                               ```python
                               def generate_items(row, col_name, parent_dfs=None):
                                   items = []
                                   if parent_dfs and 'Product' in parent_dfs and 'Transaction' in parent_dfs:
                                       products_df = parent_dfs['Product']
                                       transactions_df = parent_dfs['Transaction']
                                       # Generate items using products and transactions data
                                       # ...
                                   return items
                               ```
                              Format: {"SchemaName": {"column_name": generator_function}}
            output_format: Format to use when saving files ('csv' or 'json')
            
        Returns:
            Dictionary mapping schema names to DataFrames of generated data
        
        Example with dictionary schemas:
            schemas = {
                'Customer': {
                    'id': 'number', 
                    'name': 'text', 
                    'email': 'email'
                },
                'Order': {
                    '__foreign_keys__': {
                        'customer_id': ["Customer", "id"]
                    },
                    'id': 'number', 
                    'customer_id': 'foreign_key', 
                    'total': 'number'
                },
                'OrderItem': {
                    'id': 'number', 
                    'order_id': {
                        'type': 'foreign_key',
                        'references': {
                            'schema': 'Order',
                            'field': 'id'
                        }
                    }, 
                    'product': 'text'
                }
            }
            
            generator.generate_for_schemas(
                schemas=schemas,
                prompts={'Customer': 'Generate tech company customers'}
            )
            
        Example with file-based schemas:
            schemas = {
                'Customer': 'schemas/customer.json',
                'Order': 'schemas/order.yaml',
                'OrderItem': 'schemas/order_item.yml'
            }
            
            generator.generate_for_schemas(
                schemas=schemas
            )
        """
        # Initialize default parameters
        if prompts is None:
            prompts = {}
        if sample_sizes is None:
            sample_sizes = {}
        if custom_generators is None:
            print("No custom generators provided")
            custom_generators = {}
            
        # Load schemas from various sources
        processed_schemas = {}
        schema_metadata = {}
        schema_descriptions = {}
        
        # Dictionary to store extracted foreign keys
        schema_foreign_keys = {}
        
        for schema_name, schema_source in schemas.items():
            # Use _get_schema_info to extract schema information regardless of source type
            llm_schema, metadata, desc, extracted_fks = self._get_schema_info(schema_source)
            
            # Store processed schema and metadata
            processed_schemas[schema_name] = llm_schema
            schema_metadata[schema_name] = metadata or {}
            schema_descriptions[schema_name] = desc or f"{schema_name} data"
            
            # Debug the metadata extraction - especially for template schemas
            if metadata and '__depends_on__' in metadata:
                print(f"Found __depends_on__ for {schema_name}: {metadata['__depends_on__']}")
                
            # Store extracted foreign keys if any
            if extracted_fks:
                schema_foreign_keys[schema_name] = extracted_fks
            
        # Build dependency information from foreign key relations and explicit dependencies
        all_dependencies = {}
        extracted_foreign_keys = {}
        
        # Add foreign keys extracted from schema files
        for schema_name, fks in schema_foreign_keys.items():
            if schema_name not in extracted_foreign_keys:
                extracted_foreign_keys[schema_name] = {}
            # Add each foreign key relationship
            for fk_column, fk_info in fks.items():
                # Handle different formats of foreign key info
                if isinstance(fk_info, tuple) and len(fk_info) == 2:
                    # Already in the correct format: (parent_schema, parent_column)
                    parent_schema, parent_column = fk_info
                elif isinstance(fk_info, dict) and 'references' in fk_info:
                    # Dictionary format with 'references' key
                    ref_parts = fk_info['references'].split('.')
                    parent_schema, parent_column = ref_parts[0], ref_parts[1]
                extracted_foreign_keys[schema_name][fk_column] = (parent_schema, parent_column)
                print(f"Using schema-defined foreign key: {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
        
        # Extract all dependencies - both foreign keys and explicit __depends_on__
        for schema_name, schema_source in schemas.items():
            # Initialize the dependency list for this schema
            if schema_name not in all_dependencies:
                all_dependencies[schema_name] = []
            
            # For YAML/JSON files, we need to check the file contents for __depends_on__
            if isinstance(schema_source, str) and os.path.isfile(schema_source):
                with open(schema_source, 'r') as f:
                    if schema_source.endswith('.json') or schema_source.endswith('.yml') or schema_source.endswith('.yaml'):
                        import yaml
                        import json
                        try:
                            # Try to parse as YAML first (works for both YAML and JSON)
                            try:
                                file_content = yaml.safe_load(f)
                            except Exception:
                                # If YAML parsing fails, try JSON
                                f.seek(0)  # Reset file pointer
                                file_content = json.load(f)
                                
                            # Check for __depends_on__ in the file content
                            if file_content and isinstance(file_content, dict) and '__depends_on__' in file_content:
                                explicit_deps = file_content['__depends_on__']
                                print(f"DEBUG: Found __depends_on__ in file {schema_source}: {explicit_deps}")
                                if isinstance(explicit_deps, list):
                                    all_dependencies[schema_name].extend(explicit_deps)
                                    for dep in explicit_deps:
                                        print(f"Using explicit dependency from file: {schema_name} depends on {dep}")
                                elif isinstance(explicit_deps, str):
                                    all_dependencies[schema_name].append(explicit_deps)
                                    print(f"Using explicit dependency from file: {schema_name} depends on {explicit_deps}")
                        except Exception as e:
                            print(f"Warning: Could not check for dependencies in {schema_source}: {str(e)}")
            
            # Get schema metadata to check for explicit dependencies from the metadata dictionary
            metadata_dict = schema_metadata.get(schema_name, {})
            
            # Add explicit dependencies from __depends_on__ field
            if isinstance(metadata_dict, dict) and '__depends_on__' in metadata_dict:
                explicit_deps = metadata_dict['__depends_on__']
                print(f"DEBUG: Found __depends_on__ for {schema_name}: {explicit_deps} (type: {type(explicit_deps)})")
                if isinstance(explicit_deps, list):
                    all_dependencies[schema_name].extend(explicit_deps)
                    for dep in explicit_deps:
                        print(f"Using explicit dependency: {schema_name} depends on {dep}")
                elif isinstance(explicit_deps, str):
                    all_dependencies[schema_name].append(explicit_deps)
                    print(f"Using explicit dependency: {schema_name} depends on {explicit_deps}")
                else:
                    print(f"WARNING: Invalid __depends_on__ format for {schema_name}: {explicit_deps}")
            
            # Add foreign key dependencies
            if schema_name in extracted_foreign_keys:
                for fk_column, (parent_schema, parent_column) in extracted_foreign_keys[schema_name].items():
                    if parent_schema not in all_dependencies[schema_name]:
                        all_dependencies[schema_name].append(parent_schema)
        
        # Calculate the generation order based on all dependencies
        generation_order = list(schemas.keys())
        try:
            # Try to sort based on dependencies
            dependency_graph = self._build_dependency_graph(
                nodes=list(schemas.keys()),
                dependencies=all_dependencies
            )
            generation_order = list(nx.topological_sort(dependency_graph))
            print("\n📊 Generation order determined:")
            for i, schema in enumerate(generation_order):
                deps = all_dependencies.get(schema, [])
                if deps:
                    print(f"  {i+1}. {schema} (depends on: {', '.join(deps)})")
                else:
                    print(f"  {i+1}. {schema} (no dependencies)")
            print("")
        except Exception as e:
            print(f"Warning: Could not determine optimal generation order: {str(e)}")
            print("Using the order provided in the schemas dictionary.")
        
        # Dictionary to hold generated data
        results = {}
        
        # Store the original generators to restore them later
        original_type_generators = self.type_generators.copy()
        original_column_generators = self.column_generators.copy()
        
        try:
            # Debug the complete generation order
            print("\n🔄 DEBUG: Schema generation order:")
            for i, schema in enumerate(generation_order):
                deps = all_dependencies.get(schema, [])
                deps_str = ", ".join(deps) if deps else "none"
                print(f"  {i+1}. {schema} (depends on: {deps_str})")
            print("")
            
            # Generate data for each schema in the correct order
            for schema_name in generation_order:
                schema = processed_schemas[schema_name]
                metadata = schema_metadata[schema_name]
                description = schema_descriptions[schema_name]
                
                print(f"\nGenerating data for {schema_name} with {len(schema)} columns")
                print(f"Description: {description}")
                
                # Get the prompt and sample size for this schema
                prompt = prompts.get(schema_name, default_prompt)
                sample_size = sample_sizes.get(schema_name, default_sample_size)
                
                # Register foreign key generators for any foreign key columns
                if schema_name in extracted_foreign_keys:
                    # Group foreign keys by parent table
                    fk_by_parent = {}
                    for fk_column, (parent_schema, parent_column) in extracted_foreign_keys[schema_name].items():
                        if parent_schema not in fk_by_parent:
                            fk_by_parent[parent_schema] = []
                        fk_by_parent[parent_schema].append((fk_column, parent_column))
                    
                    # Process each parent table group
                    for parent_schema, fk_list in fk_by_parent.items():
                        if parent_schema in results:
                            parent_df = results[parent_schema]
                            
                            # Multiple columns referencing the same parent table
                            if len(fk_list) > 1:
                                print(f"Ensuring consistent foreign keys for {len(fk_list)} columns in {schema_name} referencing {parent_schema}")
                                
                                # For each row we'll generate, select a consistent parent record index
                                # This ensures all foreign keys referencing the same parent table will be from the same record
                                parent_indices = list(range(len(parent_df)))
                                if not parent_indices:
                                    print(f"⚠️ Warning: No records in {parent_schema} for foreign keys in {schema_name}")
                                    continue
                                
                                # We need a consistent set of indices across all columns in this group
                                # Create a shared state between all generators
                                shared_state = {
                                    'row_to_parent': [random.choice(parent_indices) for _ in range(sample_size)],
                                    'parent_df': parent_df,
                                    'row_cache': {}  # Cache to store row-to-parent mappings
                                }
                                
                                # Register a generator for each column that uses the shared mapping
                                for fk_column, parent_column in fk_list:
                                    # Create a generator that uses the shared state to return consistent values
                                    def create_consistent_generator(col, state, parent_col):
                                        def generator(row, col_name):
                                            # Get a stable identifier for this row
                                            # We use repr(row) as a stable key for the row
                                            row_key = repr(row)
                                            
                                            # If we haven't seen this row before, assign it a parent
                                            if row_key not in state['row_cache']:
                                                row_idx = len(state['row_cache']) % len(state['row_to_parent'])
                                                state['row_cache'][row_key] = state['row_to_parent'][row_idx]
                                                
                                            # Get the parent index for this row
                                            parent_idx = state['row_cache'][row_key]
                                            
                                            # Return the value from the parent record
                                            return state['parent_df'].iloc[parent_idx][parent_col]
                                        
                                        return generator
                                    
                                    fk_generator = create_consistent_generator(fk_column, shared_state, parent_column)
                                    
                                    print(f"Registering consistent foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
                                    self.register_generator('foreign_key', fk_generator, column_name=fk_column)
                            else:
                                # Only one column referencing this parent table, use regular random selection
                                for fk_column, parent_column in fk_list:
                                    valid_values = parent_df[parent_column].tolist()
                                    
                                    if not valid_values:
                                        print(f"⚠️ Warning: No valid values found in {parent_schema}.{parent_column} for foreign key {schema_name}.{fk_column}")
                                        continue
                                    
                                    # Create a generator that returns a random valid value
                                    values_copy = valid_values.copy()  # Make a copy to avoid reference issues
                                    fk_generator = lambda row, col, values=values_copy: random.choice(values)
                                    
                                    # Register the generator for this column
                                    print(f"Registering foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
                                    self.register_generator('foreign_key', fk_generator, column_name=fk_column)
                        else:
                            for fk_column, parent_column in fk_list:
                                print(f"⚠️ Warning: Parent schema {parent_schema} not available for foreign key {schema_name}.{fk_column}")
                
                # We'll let generate_data handle the prompt building with metadata
                # by passing the schema directly, along with the base prompt
                # This eliminates duplicated prompt-building logic
                # Use AI-based generation for meaningful data
                print(f"Creating data for {schema_name} with schema: {schema}")
                
                # Use the schema information we already extracted earlier
                llm_schema = processed_schemas[schema_name]
                metadata = schema_metadata[schema_name]
                model_description = schema_descriptions[schema_name]
                
                # Try to use the AI generation first
                try:
                    # Use the _generate_data method to generate data for this schema
                    print(f"Generating data for {schema_name} using _generate_data")
                    # Pass the already extracted schema information to avoid redundant extraction
                    df = self._generate_data(
                        table_schema=llm_schema, 
                        metadata=metadata, 
                        table_description=model_description,
                        prompt=prompt, 
                        sample_size=sample_size
                    )
                    
                    # Check if we have the requested sample size
                    if len(df) < sample_size:
                        print(f"Warning: LLM generated only {len(df)} records instead of {sample_size} for {schema_name}")
                        # We don't fill with placeholder data - we'll use what the LLM gave us
                    
                    # Truncate if we got more data than needed
                    if len(df) > sample_size:
                        df = df.iloc[:sample_size]
                        
                except Exception as e:
                    print(f"Error using AI generation for {schema_name}: {str(e)}")
                    # We don't use placeholder data - require a real LLM
                    raise Exception(f"Failed to generate data for {schema_name} using LLM: {str(e)}")
                
                # Add placeholder methods needed for AI generation
                
                # Second pass: handle foreign key fields
                for field_name, field_info in schema.items():
                    if field_name.startswith('__'):
                        continue
                        
                    field_type = field_info if isinstance(field_info, str) else field_info.get('type', 'text')
                    
                    # Now handle foreign key fields
                    if field_type.lower() == 'foreign_key':
                        # Extract references info
                        references = None
                        if isinstance(field_info, dict) and 'references' in field_info:
                            references = field_info['references']
                            parent_schema = references.get('schema')
                            parent_field = references.get('field')
                            
                            # Handle self-referential foreign keys specially
                            if parent_schema == schema_name:
                                # For self-references, we need to be careful
                                # First row should have NULL or 0 as parent (root)
                                values = [None]  # Start with NULL for first item (root)
                                
                                for key, value in yaml_content.items():
                                    # Handle special metadata fields
                                    if key.startswith('__') and key.endswith('__'):
                                        # Store metadata but don't include in schema
                                        metadata[key] = value
                                        # Extract description if available
                                        if key == '__description__':
                                            description = value
                                        # Extract foreign keys if defined
                                        elif key == '__foreign_keys__' and isinstance(value, dict):
                                            for fk_column, fk_ref in value.items():
                                                # Handle both list format [parent_table, parent_column] and string format
                                                if isinstance(fk_ref, list) and len(fk_ref) == 2:
                                                    parent_table, parent_column = fk_ref
                                                    foreign_keys[fk_column] = (parent_table, parent_column)
                                                elif isinstance(fk_ref, str):
                                                    # If string format, assume column name is 'id'
                                                    foreign_keys[fk_column] = (fk_ref, 'id')
                                                else:
                                                    print(f"Warning: Invalid foreign key format for {key}.{fk_column}")
                                        # Extract explicit dependencies
                                        elif key == '__depends_on__':
                                            # Just store it in metadata, we'll process it later
                                            print(f"Storing dependency info: {key} = {value} (type: {type(value)})")
                                    else:
                                        # Regular field - add to schema
                                        field_type = None
                                        if isinstance(value, dict) and 'type' in value:
                                            field_type = value['type']
                                        else:
                                            # Default to string if type not specified
                                            field_type = 'string'
                                        
                                        table_schema[key] = field_type
                                print(f"Created hierarchical self-references for {schema_name}.{field_name}")
                            
                            # Use existing results if the parent schema has already been processed (not self-referential)
                            elif parent_schema in results and parent_field in results[parent_schema].columns:
                                valid_ids = results[parent_schema][parent_field].tolist()
                                # Ensure we have enough IDs (repeat if necessary)
                                while len(valid_ids) < sample_size:
                                    valid_ids.extend(valid_ids)
                                # Select random IDs from the parent schema
                                df[field_name] = [random.choice(valid_ids) for _ in range(sample_size)]
                            else:
                                # Parent schema not yet processed, use placeholder IDs
                                print(f"WARNING: Parent schema {parent_schema} not yet processed. Using placeholder IDs for {field_name}")
                                df[field_name] = [random.randint(1, 1000) for _ in range(sample_size)]
                
                # Apply custom generators if any
                schema_custom_generators = custom_generators.get(schema_name, {})
                df = self._apply_custom_generators(df, schema_name, schema_custom_generators, parent_dfs=results)
                
                # Store the result
                results[schema_name] = df
                
            # Separate template schemas from structured schemas
            template_schemas = {}
            structured_results = {}
            
            for schema_name, df in results.items():
                # Check if this is a template schema by looking for template_source field
                if df is not None and 'template_source' in df.columns:
                    template_schemas[schema_name] = schemas[schema_name]
                else:
                    structured_results[schema_name] = df
            
            # Process template schemas if any
            if template_schemas and output_dir:
                # Process each template schema
                from syda.templates import TemplateProcessor
                processor = TemplateProcessor()
                
                for schema_name, df in results.items():
                    # Skip if not a template schema
                    if not ('template_source' in df.columns):
                        continue
                        
                    print(f"Processing {schema_name} templates...")
                    
                    # Create output directory for this schema
                    template_output_dir = os.path.join(output_dir, schema_name)
                    os.makedirs(template_output_dir, exist_ok=True)
                    
                    # Process each row in the dataframe
                    documents_generated = 0
                    
                    for idx, row in df.iterrows():
                        template_path = row.get('template_source')
                        input_file_type = row.get('input_file_type', '').lower()
                        output_file_type = row.get('output_file_type', '').lower()
                        
                        # Skip if missing required fields
                        if not template_path or not os.path.exists(template_path) or not input_file_type or not output_file_type:
                            print(f"Warning: Invalid template configuration for {schema_name} row {idx}")
                            continue
                            
                        # Output path for this document
                        output_path = os.path.join(template_output_dir, f"document_{idx+1}.{output_file_type}")
                        
                        try:
                            # Process the template with data
                            processor.process_template_with_data(
                                template_path=template_path,
                                data=row.to_dict(),
                                output_path=output_path,
                                input_file_type=input_file_type,
                                output_file_type=output_file_type
                            )
                            documents_generated += 1
                            print(f"✓ Successfully generated: {output_path}")
                        except Exception as e:
                            print(f"Error generating document for {schema_name} row {idx}: {str(e)}")
                    
                    print(f"Generated {documents_generated} documents for {schema_name}")
            
            # Save files if output_dir is specified
            if output_dir:
                save_dataframes(structured_results, output_dir, format=output_format)
            
            # Verify referential integrity
            print("\n🔍 Verifying referential integrity:")
            all_valid = True
            for schema_name, fk_columns in extracted_foreign_keys.items():
                for fk_column, (parent_schema, parent_column) in fk_columns.items():
                    # Skip validation if we don't have both tables
                    if schema_name not in results or parent_schema not in results:
                        print(f"  ⚠️ Cannot verify {schema_name}.{fk_column} references - missing tables")
                        continue
                        
                    # Get the values used in the foreign key column
                    fk_values = results[schema_name][fk_column].tolist()
                    # Get the valid values from the parent table
                    valid_values = results[parent_schema][parent_column].tolist()
                    
                    # Check if all foreign key values are valid
                    invalid_values = [v for v in fk_values if v not in valid_values]
                    if invalid_values:
                        print(f"  ❌ Invalid {schema_name}.{fk_column} references detected")
                        all_valid = False
                    else:
                        print(f"  ✅ All {schema_name}.{fk_column} values reference valid {parent_schema}.{parent_column}")
            
            if not all_valid:
                print("\n⚠️ Some foreign key constraints were violated. This may affect data integrity.")
            elif not extracted_foreign_keys:
                print("  ℹ️ No foreign key relationships defined in schemas")
                    
        except Exception as e:
            # Restore original generators in case of error
            self.type_generators = original_type_generators
            self.column_generators = original_column_generators
            raise e
            
        return results

    def _get_schema_info(self, schema):
        """
        Extract schema information based on the type of schema provided.
        
        Args:
            schema: Either a dictionary mapping field names to types,
                    a SQLAlchemy model class, or a path to a JSON/YAML schema file
                    
        Returns:
            Tuple of (table_schema, metadata, table_description, foreign_keys)
            
            Example return values:
                - table_schema: {'id': 'number', 'name': 'text', 'email': 'email', 'created_at': 'date'}
                - metadata: {
                    'id': {'description': 'Primary key', 'constraints': {'primary_key': True}},
                    'name': {'description': 'User full name', 'constraints': {'max_length': 100}},
                    'email': {'description': 'User email address', 'constraints': {'unique': True}},
                    'created_at': {'description': 'Account creation date'}
                }
                - table_description: "User account information for the system"
                - foreign_keys: {'user_id': ('User', 'id')}
        """
        table_schema = {}
        metadata = {}
        table_description = None
        foreign_keys = {}
        
        # Case 1: Dictionary schema
        if isinstance(schema, dict):
            # Make a copy of the schema, but filter out special fields (starting with __)
            table_schema = {k: v for k, v in schema.items() if not k.startswith('__')}
            
            # Extract table description if present using the special key
            if "__table_description__" in schema:
                table_description = schema["__table_description__"]
                
            # Extract foreign keys if present using the special key
            if "__foreign_keys__" in schema:
                # Convert lists to tuples if needed since JSON uses lists instead of tuples
                for fk_col, fk_ref in schema["__foreign_keys__"].items():
                    if isinstance(fk_ref, list) and len(fk_ref) == 2:
                        foreign_keys[fk_col] = (fk_ref[0], fk_ref[1])
                    else:
                        foreign_keys[fk_col] = fk_ref
            
            # Process each field in the schema
            for field_name, field_info in schema.items():
                # Skip special fields that start with __
                if field_name.startswith("__"):
                    continue
                    
                # Handle simple field definition (field: "type")
                if isinstance(field_info, str):
                    table_schema[field_name] = field_info
                    metadata[field_name] = {}
                # Handle complex field definition with metadata
                elif isinstance(field_info, dict):
                    # Extract the field type
                    field_type = field_info.get("type", "text")
                    table_schema[field_name] = field_type
                    
                    # Extract metadata for this field
                    field_metadata = {}
                    
                    # Add description if available
                    if "description" in field_info:
                        field_metadata["description"] = field_info["description"]
                    
                    # Extract foreign key relationships from references
                    if field_type == "foreign_key" and "references" in field_info:
                        references = field_info["references"]
                        if "schema" in references and "field" in references:
                            # Add to foreign keys dictionary
                            foreign_keys[field_name] = (references["schema"], references["field"])
                    
                    # Add constraints if available
                    constraints = {}
                    
                    # Check for length constraints directly in the field definition
                    if "length" in field_info:
                        if not "constraints" in field_metadata:
                            field_metadata["constraints"] = {}
                        field_metadata["constraints"]["length"] = field_info["length"]
                    
                    if "max_length" in field_info:
                        if not "constraints" in field_metadata:
                            field_metadata["constraints"] = {}
                        field_metadata["constraints"]["max_length"] = field_info["max_length"]
                    
                    if "min_length" in field_info:
                        if not "constraints" in field_metadata:
                            field_metadata["constraints"] = {}
                        field_metadata["constraints"]["min_length"] = field_info["min_length"]
                    
                    # Add constraints if explicitly defined in constraints section
                    if "constraints" in field_info:
                        if not "constraints" in field_metadata:
                            field_metadata["constraints"] = {}
                        # Add all constraints from the constraints dict
                        for k, v in field_info["constraints"].items():
                            field_metadata["constraints"][k] = v
                    
                    metadata[field_name] = field_metadata
            
        # Case 2: SQLAlchemy model - check for __table__ attribute which all SQLAlchemy models have
        elif isinstance(schema, type) and hasattr(schema, '__table__'):
            table_schema, metadata, table_description = sqlalchemy_model_to_schema(schema)
            # SQLAlchemy foreign keys are handled differently, return empty dict for now
            foreign_keys = {}
        
        # Case 3: Path to JSON schema file
        elif isinstance(schema, str) and (schema.endswith('.json') or schema.endswith('.schema')):
            with open(schema, 'r') as f:
                schema_dict = json.load(f)
                # Process the loaded dictionary recursively
                return self._get_schema_info(schema_dict)
                
        # Case 4: Path to YAML schema file
        elif isinstance(schema, str) and (schema.endswith('.yml') or schema.endswith('.yaml')):
            try:
                import yaml
                with open(schema, 'r') as f:
                    schema_dict = yaml.safe_load(f)
                    # Process the loaded dictionary recursively
                    return self._get_schema_info(schema_dict)
            except ImportError:
                raise ImportError("PyYAML is required for YAML schema support. Install it with 'pip install pyyaml'.")
            except Exception as e:
                raise ValueError(f"Error loading YAML schema file {schema}: {str(e)}")
        return table_schema, metadata, table_description, foreign_keys

    def _build_prompt(self, table_schema, metadata, table_description, primary_key_fields, prompt, sample_size):
        """
        Build a structured prompt for the LLM to generate data.
        
        Args:
            table_schema: Dictionary mapping field names to types
            metadata: Dictionary with field metadata including descriptions and constraints
            table_description: Optional description of the table
            primary_key_fields: List of primary key field names
            prompt: Base prompt text
            sample_size: Number of records to generate
            
        Returns:
            Formatted prompt for the LLM
        """
        # Create a list to hold field descriptions
        field_descriptions = []
        
        # Add each field with its type and constraints
        for field_name, field_type in table_schema.items():
            # Start with the basic field info
            field_desc = f"- {field_name}: {field_type}"
            
            # Add constraints and descriptions from metadata if available
            if field_name in metadata:
                field_meta = metadata[field_name]
                
                # Add description if available
                if 'description' in field_meta and field_meta['description']:
                    field_desc += f" ({field_meta['description']})"
                    
                # Add constraints if available
                if 'constraints' in field_meta:
                    constraints = field_meta['constraints']
                    constraint_parts = []
                    
                    # Add primary key constraint
                    if 'primary_key' in constraints and constraints['primary_key']:
                        constraint_parts.append("primary_key: True")
                        
                    # Add unique constraint
                    if 'unique' in constraints and constraints['unique']:
                        constraint_parts.append("unique: True")
                        
                    # Add length constraints - these are important for text fields
                    if 'length' in constraints:
                        constraint_parts.append(f"length: {constraints['length']}")
                    if 'max_length' in constraints:
                        constraint_parts.append(f"max_length: {constraints['max_length']}")
                    if 'min_length' in constraints:
                        constraint_parts.append(f"min_length: {constraints['min_length']}")
                    
                    # Add not_null constraint
                    if 'not_null' in constraints and constraints['not_null']:
                        constraint_parts.append("not_null: True")
                    
                    # Add foreign key constraint
                    if 'foreign_key_to' in constraints:
                        constraint_parts.append(f"foreign_key_to: {constraints['foreign_key_to']}")
                    
                    # Add any other constraints
                    for k, v in constraints.items():
                        if k not in ['primary_key', 'unique', 'length', 'max_length', 'min_length', 'not_null', 'foreign_key_to']:
                            constraint_parts.append(f"{k}: {v}")
                    
                    # Add constraints to field description
                    if constraint_parts:
                        constraint_str = ", ".join(constraint_parts)
                        field_desc += f" [{constraint_str}]"
            
            field_descriptions.append(field_desc)
        
        # Start with the basic instruction
        full_prompt = f"Generate {sample_size} records JSON objects with these fields:\n"
        full_prompt += "\n".join(field_descriptions)
        
        # Add the description
        if prompt and prompt != "Generate synthetic data":
            full_prompt += f"\nDescription: {prompt}"
        
        # Include table description if available
        if table_description:
            full_prompt += f"\nTable Description: {table_description}"
            
        print(f"Full Prompt: {full_prompt}")
        
        return full_prompt
            

    def _generate_data_with_llm(self, llm_schema, full_prompt, sample_size):
        """
        Generate data using the LLM based on the schema and prompt.
        
        Args:
            llm_schema: Dictionary mapping field names to types
            full_prompt: Complete prompt to send to the LLM
            sample_size: Number of records to generate
            
        Returns:
            DataFrame of generated data
            
        Raises:
            ValueError: If LLM data generation fails
        """
        # Build Pydantic model for parsing with proper type mapping
        def get_python_type(field_type):
            """Map schema types to Python types."""
            if isinstance(field_type, str):
                type_map = {
                    'integer': int,
                    'int': int,
                    'float': float,
                    'number': float,
                    'boolean': bool,
                    'bool': bool,
                    'array': list,
                    'object': dict
                }
                return type_map.get(field_type.lower(), str)
            elif isinstance(field_type, dict) and 'type' in field_type:
                return get_python_type(field_type['type'])
            return str  # Default to string for unknown types
        
        fields = {}
        for col, field_info in llm_schema.items():
            if isinstance(field_info, dict) and 'type' in field_info:
                fields[col] = (get_python_type(field_info['type']), ...)
            elif isinstance(field_info, str):
                fields[col] = (get_python_type(field_info), ...)
            else:
                fields[col] = (str, ...)  # Default to string
        DynamicInstructorModel = create_model("DynamicModel", **fields)

        print(f"Generating data using {self.model_config.provider}/{self.model_config.model_name}...")
        
        # Get model kwargs with proper API key handling
        model_kwargs = self.llm_client.get_model_kwargs()  # This ensures api_keys are not passed directly
        
        # Ensure the model name is always included
        if 'model' not in model_kwargs:
            model_kwargs['model'] = self.model_config.model_name
        
        try:
            print(f"Full Prompt: {full_prompt}")
            # Call the LLM through instructor's unified interface
            ai_objs = self.client.chat.completions.create(
                response_model=List[DynamicInstructorModel],
                messages=[{"role": "user", "content": full_prompt}],
                **model_kwargs,
            )
            
            if not ai_objs:
                raise ValueError("No objects returned from LLM call")
            
            # Convert objects to DataFrame
            records = [obj.model_dump() for obj in ai_objs]
            
            # Debug log
            if not records:
                raise ValueError("No records extracted from LLM response")
            else:
                print(f"✓ Successfully generated {len(records)} records")
                print(f"  Fields in first record: {list(records[0].keys()) if records else 'None'}")
            
            # Create DataFrame from records
            df = pd.DataFrame(records)
            
            # Ensure all expected columns are present
            for col in llm_schema.keys():
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in generated data. Data generation failed to produce the expected schema.")
            
            # If no data was returned, fail
            if df.empty:
                raise ValueError("Empty DataFrame returned from data generation")
                
            # Validate DataFrame structure
            print(f"DataFrame shape: {df.shape} with columns: {list(df.columns)}")
            
            return df
                
        except Exception as e:
            raise ValueError(f"Error generating data: {str(e)}")

    def _apply_custom_generators(self, df, model_name, custom_generators, parent_dfs=None):
        """
        Apply custom generators to the generated data.
        
        Args:
            df: DataFrame to apply generators to
            model_name: Name of the model being processed
            custom_generators: Dictionary of custom generators for the model
            parent_dfs: Optional dictionary of previously generated dataframes
            
        Returns:
            DataFrame with custom generators applied
        """
        if not custom_generators:
            return df
            
        if parent_dfs is None:
            parent_dfs = {}
            
        for col_name, gen_func in custom_generators.items():
            if col_name in df.columns:
                print(f"Applying custom generator for {model_name}.{col_name}")
                # Check function signature to maintain backward compatibility
                import inspect
                sig = inspect.signature(gen_func)
                if len(sig.parameters) >= 3:  # If the function accepts at least 3 parameters
                    df[col_name] = df.apply(lambda row: gen_func(row, col_name, parent_dfs), axis=1)
                else:  # Backward compatibility for existing functions
                    df[col_name] = df.apply(lambda row: gen_func(row, col_name), axis=1)
                
        return df
    
    def _apply_type_generators(self, df, llm_schema):
        """
        Apply custom type-based and column-specific generators to the data.
        
        Args:
            df: DataFrame to apply generators to
            llm_schema: Dictionary mapping field names to types
            
        Returns:
            DataFrame with generators applied
        """
        for col in llm_schema.keys():
            # Ensure the column exists in the DataFrame
            if col not in df.columns:
                print(f"⚠️ Adding missing column '{col}' before applying generators")
                df[col] = [f"auto_{col}_{i}" for i in range(len(df))]
            
            # Custom generator for this specific column takes priority
            if col in self.column_generators:
                print(f"Applying custom generator for column '{col}'")
                df[col] = df.apply(lambda row: self.column_generators[col](row, col), axis=1)
            # Otherwise, try type-based generator if available
            elif col in llm_schema and llm_schema[col].lower() in self.type_generators:
                print(f"Applying type-based generator for column '{col}'")
                df[col] = df.apply(lambda row: self.type_generators[llm_schema[col].lower()](row, col), axis=1)
                
        return df
    
    def _convert_column_types(self, df, llm_schema):
        """
        Convert DataFrame columns to appropriate types based on schema.
        
        Args:
            df: DataFrame to convert column types for
            llm_schema: Dictionary mapping field names to types
            
        Returns:
            DataFrame with converted column types
        """
        for col in df.columns:
            if col in llm_schema:
                dtype = llm_schema[col].lower()
                if dtype == 'number':
                    try:
                        # First try integer conversion
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    except:
                        try:
                            # Fall back to float if integer fails
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass  # Keep as string if conversion fails
                elif dtype == 'boolean':
                    try:
                        # Convert various boolean representations
                        df[col] = df[col].map({'True': True, 'true': True, 'FALSE': False, 'false': False, '1': True, '0': False})
                    except:
                        pass
        return df
    
    # Method _save_output has been moved to output.py module

    def _generate_data(self, table_schema: Dict[str, str],
                     metadata: Dict[str, Dict],
                     table_description: Optional[str] = None,
                     prompt: str = "Generate synthetic data",
                     sample_size: int = 10) -> pd.DataFrame:
        """
        Generate synthetic data based on schema using AI.
        
        Args:
            table_schema: Dictionary mapping field names to types (e.g., {'name': 'text', 'age': 'number'})
            metadata: Dictionary with additional info about fields
            table_description: Optional description of the table to guide generation
            prompt: Prompt for the AI model
            sample_size: Number of samples to generate
            
        Returns:
            DataFrame with generated data
            
        Raises:
            ValueError: If the data generation fails or produces invalid results
        """
        
        print("Using extracted schema information")
        print("table_schema", table_schema)
        
        # Identify primary key fields
        primary_key_fields = []
        for col, col_meta in metadata.items():
            if 'constraints' in col_meta and 'primary_key' in col_meta['constraints']:
                primary_key_fields.append(col)
        
        # Build prompt for LLM
        full_prompt = self._build_prompt(table_schema, metadata, table_description, 
                                      primary_key_fields, prompt, sample_size)
        
        # Generate data using LLM
        df = self._generate_data_with_llm(table_schema, full_prompt, sample_size)
        
        # Apply type-based generators
        df = self._apply_type_generators(df, table_schema)
        
        # Convert column types based on schema
        df = self._convert_column_types(df, table_schema)
        
        # Apply custom generators if they exist
        if hasattr(self, 'model_custom_generators') and self.model_custom_generators:
            df = self._apply_custom_generators(df, "model", self.model_custom_generators, parent_dfs={})
            
        # Apply column-specific generators
        for col_name in df.columns:
            if col_name in self.column_generators:
                print(f"Applying custom generator for {col_name}")
                df[col_name] = df.apply(lambda row: self.column_generators[col_name](row, col_name), axis=1)
        
        return df
