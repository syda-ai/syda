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
from .utils import (
    sqlalchemy_model_to_schema, 
    extract_sqlalchemy_relationships,
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

    def _apply_custom_generators(self, df, model_name, model_custom_generators):
        """
        Apply custom generators to a DataFrame based on model and column specifications.
        
        Args:
            df: DataFrame to modify
            model_name: Name of the model
            model_custom_generators: Dict mapping column names to generator functions
            
        Returns:
            pd.DataFrame: Modified DataFrame with custom values
        """
        if not model_custom_generators:
            return df
            
        num_custom_generators = len(model_custom_generators)
        if num_custom_generators > 0:
            print(f"Found {num_custom_generators} custom generators for {model_name}")
            for col_name, generator_fn in model_custom_generators.items():
                if col_name in df.columns:
                    print(f"Applying custom generator for {model_name}.{col_name}")
                    # Apply the custom generator row by row
                    for i in range(len(df)):
                        row = df.iloc[i].copy()
                        df.at[i, col_name] = generator_fn(row, col_name)
        return df

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
        custom_generators: Optional[Dict[str, Dict[str, Callable]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for multiple relational SQLAlchemy models with automatic 
        dependency resolution based on foreign key relationships.
        
        This function:
        1. Analyzes SQLAlchemy model dependencies (which SQLAlchemy models reference others)
        2. Automatically determines the correct order to generate data
        3. Handles foreign key relationships between SQLAlchemy models
        4. Applies custom generators where registered
        
        Args:
            sqlalchemy_models: A list of SQLAlchemy model classes, a single SQLAlchemy model class, 
                    or a string pattern to match class names
            prompts: Optional dictionary mapping model names to custom prompts
            sample_sizes: Optional dictionary mapping model names to sample sizes
            output_dir: Optional directory to save CSV files (one per model)
            default_sample_size: Default number of records if not specified in sample_sizes
            default_prompt: Default prompt if not specified in prompts
            custom_generators: Optional dictionary specifying custom generators for SQLAlchemy models and columns.
                               Format: {"ModelName": {"column_name": generator_function}}
            
        Returns:
            Dictionary mapping model names to DataFrames of generated data
            
        Example:
            # Define custom generators for specific columns in specific SQLAlchemy models
            custom_gens = {
                "Customer": {
                    "status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"])
                },
                "Product": {
                    "price": lambda row, col: round(random.uniform(50, 500), 2)
                }
            }
            
            results = generator.generate_for_sqlalchemy_models(
                sqlalchemy_models=[Customer, Order, OrderItem, Product],
                prompts={"Customer": "Generate tech companies"},
                sample_sizes={"Customer": 10, "Order": 30},
                custom_generators=custom_gens
            )
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
            
        # Extract schema and metadata information for each model
        model_info = {}
        model_names = []
        model_dependencies = {}
        
        # Create model to schema mapping
        for model_class in sqlalchemy_models:
            model_name = model_class.__name__
            model_names.append(model_name)
            
            # Extract schema, metadata, and docstring from the SQLAlchemy model
            schema, metadata, docstring = sqlalchemy_model_to_schema(model_class)
            
            # Store model info
            model_info[model_name] = {
                'model_class': model_class,
                'schema': schema,
                'metadata': metadata,
                'docstring': docstring,
                'references': set(),  # Will contain names of models this model references
                'referenced_by': set()  # Will contain names of models that reference this model
            }
        
        # Find foreign key dependencies between models
        table_to_model_name = {m.__table__.name: m.__name__ for m in sqlalchemy_models}
        
        for model_name, info in model_info.items():
            model_class = info['model_class']
            dependencies = []
            
            # Search for foreign keys and extract dependencies
            for column in model_class.__table__.columns:
                for fk in column.foreign_keys:
                    target_table = fk.column.table.name
                    if target_table in table_to_model_name:
                        target_model = table_to_model_name[target_table]
                        if target_model != model_name:  # Avoid self-references
                            dependencies.append(target_model)
                            # Update references info
                            info['references'].add(target_model)
                            model_info[target_model]['referenced_by'].add(model_name)
            
            # Store this model's dependencies
            if dependencies:
                model_dependencies[model_name] = dependencies
        
        # Use the shared dependency graph builder
        G = self._build_dependency_graph(nodes=model_names, dependencies=model_dependencies)
        
        # Determine generation order using topological sort
        generation_order = list(nx.topological_sort(G))
        
        # Dictionary to hold generated data
        results = {}
        
        # Store the original generators to restore them later
        original_type_generators = self.type_generators.copy()
        original_column_generators = self.column_generators.copy()
        
        try:
            # Generate data for each model in the correct order
            for model_name in generation_order:
                print(f"\nGenerating data for {model_name} with {len(model_info[model_name]['schema'])} columns")
                print(f"Schema: {model_info[model_name]['schema']}")
                
                # Get model info
                model_class = model_info[model_name]['model_class']
                schema = model_info[model_name]['schema']
                
                # Get the prompt and sample size for this model
                prompt = prompts.get(model_name, default_prompt)
                sample_size = sample_sizes.get(model_name, default_sample_size)
                print(f"Using prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Using prompt: {prompt}")
                
                # Generate the data
                # Extract schema information first
                llm_schema, metadata, model_description, _ = self._get_schema_info(model_class)
                
                # Use the renamed _generate_data method with pre-extracted schema info
                df = self._generate_data(
                    table_schema=llm_schema, 
                    metadata=metadata, 
                    table_description=model_description,
                    prompt=prompt, 
                    sample_size=sample_size
                )
                
                # Handle any missing columns
                df = self._handle_missing_columns(df, model_name, schema, None, custom_generators)
                
                # Apply custom generators
                model_custom_generators = custom_generators.get(model_name, {})
                df = self._apply_custom_generators(df, model_name, model_custom_generators)
                
                # Store the generated data
                results[model_name] = df
                
                # After generating a model, register foreign key generators for any models that reference it
                # and store valid IDs for later use
                for dependent_model, info in model_info.items():
                    if model_name in info['references']:
                        # This model references the model we just generated
                        # Find the foreign key columns that reference this model
                        dependent_model_class = model_info[dependent_model]['model_class']
                        # Create a column-specific foreign key generator that samples from the IDs we just generated
                        for col_name, col_type in model_info[dependent_model]['schema'].items():
                            if col_type == 'foreign_key':
                                inspector = inspect(dependent_model_class)
                                for fk in inspector.columns[col_name].foreign_keys:
                                    target_fullname = fk.target_fullname
                                    target_table = target_fullname.split('.')[0]
                                    target_column = target_fullname.split('.')[1]
                                    # If this column references the current model
                                    if model_info[model_name]['model_class'].__table__.name == target_table:
                                        # Capture the valid IDs from the parent model
                                        # Ensure we use the actual primary key values
                                        valid_ids = df[target_column].tolist() if target_column in df.columns else df['id'].tolist()
                                        
                                        # Save these valid IDs for later validation
                                        if 'fk_values' not in model_info[dependent_model]:
                                            model_info[dependent_model]['fk_values'] = {}
                                        model_info[dependent_model]['fk_values'][col_name] = valid_ids
                                        
                                        # Create a stable generator function that captures valid_ids
                                        valid_ids_copy = valid_ids.copy()  # Make a copy to avoid reference issues
                                        fk_generator = lambda row, col, ids=valid_ids_copy: random.choice(ids)
                                        
                                        # Register this generator for the specific column
                                        print(f"Registering foreign key generator for {dependent_model}.{col_name} -> {model_name}")
                                        self.register_generator('foreign_key', fk_generator, column_name=col_name)
                
                # Save to file if output_dir is specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{model_name.lower()}.csv")
                    df.to_csv(output_path, index=False)
                    
        except Exception as e:
            # Restore original generators in case of error
            self.type_generators = original_type_generators
            self.column_generators = original_column_generators
            raise e
            
        return results
        
    def generate_for_schemas(
        self,
        schemas: Dict[str, Union[Dict[str, str], str]],
        prompts: Optional[Dict[str, str]] = None,
        sample_sizes: Optional[Dict[str, int]] = None,
        output_dir: Optional[str] = None,
        default_sample_size: int = 10,
        default_prompt: str = "Generate synthetic data",
        custom_generators: Optional[Dict[str, Dict[str, Callable]]] = None
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
            output_dir: Optional directory to save CSV files (one per schema)
            default_sample_size: Default number of records if not specified in sample_sizes
            default_prompt: Default prompt if not specified in prompts
            custom_generators: Optional dictionary specifying custom generators for schemas and columns
                              Format: {"SchemaName": {"column_name": generator_function}}
            
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
            
            # Store extracted foreign keys if any
            if extracted_fks:
                schema_foreign_keys[schema_name] = extracted_fks
            
        # Build dependency information from foreign key relations
        schema_dependencies = {}
        extracted_foreign_keys = {}
        
        # Add foreign keys extracted from schema files
        for schema_name, fks in schema_foreign_keys.items():
            if schema_name not in extracted_foreign_keys:
                extracted_foreign_keys[schema_name] = {}
            # Add each foreign key relationship
            for fk_column, (parent_schema, parent_column) in fks.items():
                extracted_foreign_keys[schema_name][fk_column] = (parent_schema, parent_column)
                print(f"Using schema-defined foreign key: {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
        
        # Extract dependencies from foreign key mappings
        for child_schema, fk_columns in extracted_foreign_keys.items():
            if child_schema not in processed_schemas:
                raise ValueError(f"Schema {child_schema} referenced in foreign keys but not found in schemas")
            
            # Collect all parent schemas for this child schema
            dependencies = []
            for fk_column, (parent_schema, parent_column) in fk_columns.items():
                if parent_schema not in processed_schemas:
                    raise ValueError(f"Parent schema {parent_schema} referenced in foreign keys but not found in schemas")
                
                # Add parent as a dependency
                dependencies.append(parent_schema)
            
            # Store this schema's dependencies
            if dependencies:
                schema_dependencies[child_schema] = dependencies
        
        # Use the shared dependency graph builder
        G = self._build_dependency_graph(
            nodes=list(processed_schemas.keys()),
            dependencies=schema_dependencies
        )
        
        # Determine generation order using topological sort
        try:
            generation_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise ValueError("Circular dependencies detected between schemas. Cannot determine generation order.")
            
        # Dictionary to hold generated data
        results = {}
        
        # Store the original generators to restore them later
        original_type_generators = self.type_generators.copy()
        original_column_generators = self.column_generators.copy()
        
        try:
            # Generate data for each schema in the correct order
            for schema_name in generation_order:
                schema = processed_schemas[schema_name]
                metadata = schema_metadata[schema_name]
                description = schema_descriptions[schema_name]
                
                print(f"\nGenerating data for {schema_name} with {len(schema)} columns")
                print(f"Schema: {schema}")
                print(f"Description: {description}")
                
                # Get the prompt and sample size for this schema
                prompt = prompts.get(schema_name, default_prompt)
                sample_size = sample_sizes.get(schema_name, default_sample_size)
                print(f"Using prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Using prompt: {prompt}")
                
                # Register foreign key generators for any foreign key columns
                if schema_name in extracted_foreign_keys:
                    for fk_column, (parent_schema, parent_column) in extracted_foreign_keys[schema_name].items():
                        if parent_schema in results:
                            # Get the valid IDs from the parent schema
                            valid_ids = results[parent_schema][parent_column].tolist()
                            
                            if not valid_ids:
                                print(f"⚠️ Warning: No valid IDs found in {parent_schema}.{parent_column} for foreign key {schema_name}.{fk_column}")
                                continue
                                
                            # Create a generator that returns a random valid ID
                            valid_ids_copy = valid_ids.copy()  # Make a copy to avoid reference issues
                            fk_generator = lambda row, col, ids=valid_ids_copy: random.choice(ids)
                            
                            # Register the generator for this column
                            print(f"Registering foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
                            self.register_generator('foreign_key', fk_generator, column_name=fk_column)
                        else:
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
                                
                                # Then for the rest, reference existing rows
                                for i in range(1, sample_size):
                                    # For each item, reference one of the previous items
                                    if i > 1:
                                        # Choose a random ID from existing rows
                                        # But ensure we don't create cycles (limit parent options)
                                        valid_parent_idx = random.randint(0, i-1)
                                        if valid_parent_idx == 0:
                                            values.append(None)  # Some items are also roots
                                        else:
                                            parent_id = df['id'].iloc[valid_parent_idx]
                                            values.append(parent_id)
                                    else:
                                        # Second item can reference the first or be null
                                        if random.choice([True, False]):
                                            values.append(df['id'].iloc[0])
                                        else:
                                            values.append(None)
                                
                                df[field_name] = values
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
                df = self._apply_custom_generators(df, schema_name, schema_custom_generators)
                
                # Store the result
                results[schema_name] = df
                
            # Save files if output_dir is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                for schema_name, df in results.items():
                    output_path = os.path.join(output_dir, f"{schema_name.lower()}.csv")
                    df.to_csv(output_path, index=False)
            
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
        # Build Pydantic model for parsing
        fields = {col: (str, ...) for col in llm_schema}
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

    def _apply_custom_generators(self, df, model_name, custom_generators):
        """
        Apply custom generators to the generated data.
        
        Args:
            df: DataFrame to apply generators to
            model_name: Name of the model being processed
            custom_generators: Dictionary of custom generators for the model
            
        Returns:
            DataFrame with custom generators applied
        """
        if not custom_generators:
            return df
            
        for col_name, gen_func in custom_generators.items():
            if col_name in df.columns:
                print(f"Applying custom generator for {model_name}.{col_name}")
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
    
    def _save_output(self, df, output_path):
        """
        Save the generated data to a file if an output path is provided.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the data to
            
        Returns:
            Path to the saved file
            
        Raises:
            ValueError: If the data cannot be saved
        """
        if not output_path:
            return None
            
        # Verify we have valid data to write
        if df.empty or len(df.columns) == 0:
            raise ValueError(
                "Failed to generate valid data. The resulting DataFrame is empty or has no columns. "
                "This could be due to an issue with the AI model response or schema definition. "
                "Check your schema, model settings, and API keys."
            )
            
        # Write the file if data is valid
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
            print(f"✓ Successfully wrote {len(df)} rows to {output_path}")
            return output_path
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records')
            print(f"✓ Successfully wrote {len(df)} rows to {output_path}")
            return output_path
        else:
            # Default to CSV if no extension is recognized
            csv_path = f"{output_path}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Successfully wrote {len(df)} rows to {csv_path}")
            return csv_path

    def _generate_data(self, table_schema: Dict[str, str],
                     metadata: Dict[str, Dict],
                     table_description: Optional[str] = None,
                     prompt: str = "Generate synthetic data",
                     sample_size: int = 10, 
                     output_path: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        Generate synthetic data based on schema using AI.
        
        Args:
            table_schema: Dictionary mapping field names to types (e.g., {'name': 'text', 'age': 'number'})
            metadata: Dictionary with field metadata including descriptions and constraints
            table_description: Optional description of the table/entity
            prompt: Description of what kind of data to generate
            sample_size: Number of records to generate
            output_path: Optional path to save generated data (csv or json)
            
        Returns:
            pandas DataFrame of generated data, or path to output file if output_path provided
            
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
            df = self._apply_custom_generators(df, "model", self.model_custom_generators)
            
        # Apply column-specific generators
        for col_name in df.columns:
            if col_name in self.column_generators:
                print(f"Applying custom generator for {col_name}")
                df[col_name] = df.apply(lambda row: self.column_generators[col_name](row, col_name), axis=1)
        
        # Save output if path provided and return
        saved_path = self._save_output(df, output_path)
        if saved_path:
            return saved_path
            
        return df
