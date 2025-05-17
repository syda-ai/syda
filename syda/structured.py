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

    def _build_model_dependency_graph(self, sqlalchemy_models):
        """
        Build a directed graph of model dependencies based on foreign key relationships.
        
        Args:
            sqlalchemy_models: List of SQLAlchemy model classes
            
        Returns:
            tuple: (G, model_info) where G is a networkx DiGraph and model_info is a dict with model metadata
        """
        # Create model dependency graph
        G = nx.DiGraph()
        
        # Model info dictionary to store info about each model
        model_info = {}
            
        # Add all models to the graph and store their schemas
        for model_class in sqlalchemy_models:
            model_name = model_class.__name__
            G.add_node(model_name)
            
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
        
        # Analyze relationships between models
        for model_name, info in model_info.items():
            model_class = info['model_class']
            schema = info['schema']
            
            # For each foreign key column, determine the target model
            for col_name, col_type in schema.items():
                if col_type == 'foreign_key':
                    # Get the target model for this foreign key
                    # This requires inspecting the SQLAlchemy model
                    inspector = inspect(model_class)
                    for fk in inspector.columns[col_name].foreign_keys:
                        target_fullname = fk.target_fullname
                        target_table = target_fullname.split('.')[0]
                        
                        # Find the model class for this table
                        target_model_name = None
                        for t_name, t_info in model_info.items():
                            if t_info['model_class'].__table__.name == target_table:
                                target_model_name = t_name
                                break
                        
                        if target_model_name:
                            # Add directed edge from target to this model
                            # (because target must be generated before this model)
                            G.add_edge(target_model_name, model_name)
                            
                            # Update references info
                            model_info[model_name]['references'].add(target_model_name)
        
        return G, model_info

    def _determine_generation_order(self, G):
        """
        Determine the order in which models should be generated based on their dependencies.
        
        Args:
            G: NetworkX DiGraph representing model dependencies
            
        Returns:
            list: Model names in the order they should be generated
        """
        try:
            return list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Handle cycles in dependency graph
            raise ValueError("Circular dependencies detected between SQLAlchemy models. Cannot determine generation order.")

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
                               where generator_function is a callable that takes (row: pd.Series, col_name: str)
                               and returns a generated value.
            
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
            
        # Build model dependency graph
        G, model_info = self._build_model_dependency_graph(sqlalchemy_models)
        
        # Determine generation order
        generation_order = self._determine_generation_order(G)
        
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
                df = self.generate_data(model_class, prompt, sample_size)
                
                # Handle any missing columns
                df = self._handle_missing_columns(df, model_name, schema, None, custom_generators)
                
                # Apply custom generators
                model_custom_generators = custom_generators.get(model_name, {})
                df = self._apply_custom_generators(df, model_name, model_custom_generators)
                
                # Store the generated data
                results[model_name] = df
                
                # Save to file if output_dir is specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{model_name.lower()}.csv")
                    df.to_csv(output_path, index=False)
                    
        except Exception as e:
            # Restore original generators in case of error
            self.type_generators = original_generators['type']
            self.column_generators = original_generators['column']
            raise e
            
        return results

    def _get_schema_info(self, schema):
        """
        Extract schema information based on the type of schema provided.
        
        Args:
            schema: Either a dictionary mapping field names to types,
                   a SQLAlchemy model class, or a path to a JSON schema file
                   
        Returns:
            Tuple of (llm_schema, metadata, model_description)
        """
        llm_schema = {}
        metadata = {}
        model_description = None
        
        # Case 1: Dictionary schema
        if isinstance(schema, dict):
            llm_schema = schema
        
        # Case 2: SQLAlchemy model - check for __table__ attribute which all SQLAlchemy models have
        elif isinstance(schema, type) and hasattr(schema, '__table__'):
            llm_schema, metadata, model_description = sqlalchemy_model_to_schema(schema)
        
        # Case 3: Path to JSON schema file
        elif isinstance(schema, str) and (schema.endswith('.json') or schema.endswith('.schema')):
            with open(schema, 'r') as f:
                llm_schema = json.load(f)
                
        return llm_schema, metadata, model_description
    
    def _build_prompt(self, llm_schema, metadata, model_description, primary_key_fields, prompt, sample_size):
        """
        Build a structured prompt for the LLM to generate data.
        
        Args:
            llm_schema: Dictionary mapping field names to types
            metadata: Dictionary of metadata for each field
            model_description: Optional description of the model
            primary_key_fields: List of primary key fields
            prompt: User-provided description of data to generate
            sample_size: Number of records to generate
            
        Returns:
            Formatted prompt string for the LLM
        """
        lines = [f"Generate {sample_size} records JSON objects with these fields:"]
        
        # Add model description if available
        if model_description:
            lines.append(f"Model Description: {model_description}")
            
        # Highlight primary key fields
        if primary_key_fields:
            lines.append(f"IMPORTANT: Always include the primary key field(s): {', '.join(primary_key_fields)}")
            
        # Add each field with type and metadata
        for col, dtype in llm_schema.items():
            # Style primary keys to make them stand out
            if col in primary_key_fields:
                field_desc = f"- {col}: {dtype} (PRIMARY KEY)"
            else:
                field_desc = f"- {col}: {dtype}"
            
            # Add field metadata if available
            if col in metadata:
                col_meta = metadata[col]
                
                # Add description
                if 'description' in col_meta:
                    field_desc += f" - {col_meta['description']}"
                
                # Add constraints
                if 'constraints' in col_meta:
                    # Filter out primary_key as we already highlighted it
                    other_constraints = [c for c in col_meta['constraints'] if c != 'primary_key']
                    if other_constraints:
                        constraints = ", ".join(other_constraints)
                        field_desc += f" ({constraints})"
                
                # Add length for string fields
                if 'length' in col_meta and dtype.lower() == 'text':
                    field_desc += f" (max length: {col_meta['length']})"
                    
                # Add default value
                if 'default' in col_meta:
                    field_desc += f" (default: {col_meta['default']})"
                    
            lines.append(field_desc)
        
        # Add user-provided prompt
        lines.append(f"Description: {prompt}")
        return "\n".join(lines)
    
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
    
    def _handle_missing_columns(self, df, model_name, schema, field_generators=None, custom_generators=None):
        """
        Handle any missing columns in the DataFrame by adding default values.
        
        Args:
            df: DataFrame to check for missing columns
            model_name: Name of the model being processed
            schema: Schema of the model
            field_generators: Optional dictionary of field generators
            custom_generators: Optional dictionary of custom generators
            
        Returns:
            DataFrame with missing columns added
        """
        for col_name in schema.keys():
            if col_name not in df.columns:
                print(f"⚠️ Adding missing column '{col_name}' in model {model_name}")
                df[col_name] = [f"default_{col_name}_{i}" for i in range(len(df))]
                
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

    def generate_data(self, schema_dict: Optional[Dict[str, str]] = None,
                     sqlalchemy_model: Optional[Type] = None,
                     schema_file_path: Optional[str] = None,
                     prompt: str = "Generate synthetic data",
                     sample_size: int = 10, 
                     output_path: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        Generate synthetic data based on schema using AI.
        
        Args:
            schema_dict: Dictionary mapping field names to types (e.g., {'name': 'text', 'age': 'number'})
            sqlalchemy_model: SQLAlchemy model class
            schema_file_path: Path to a JSON schema file
            prompt: Description of what kind of data to generate
            sample_size: Number of records to generate
            output_path: Optional path to save generated data (csv or json)
            
        Returns:
            pandas DataFrame of generated data, or path to output file if output_path provided
            
        Raises:
            ValueError: If the data generation fails or produces invalid results
        """
        
        # Determine which schema source to use
        schema = None
        if schema_dict is not None:
            schema = schema_dict
        elif sqlalchemy_model is not None:
            schema = sqlalchemy_model
        elif schema_file_path is not None:
            schema = schema_file_path
        else:
            raise ValueError("You must provide either schema_dict, sqlalchemy_model, or schema_file_path")
        # Extract schema information
        llm_schema, metadata, model_description = self._get_schema_info(schema)
        print("schema", schema)
        print("llm_schema", llm_schema)
        # Identify primary key fields
        primary_key_fields = []
        for col, col_meta in metadata.items():
            if 'constraints' in col_meta and 'primary_key' in col_meta['constraints']:
                primary_key_fields.append(col)
        
        # Generate DataFrame, either from LLM or empty template
        df = None
        if llm_schema:
            # Build prompt for LLM
            full_prompt = self._build_prompt(llm_schema, metadata, model_description, 
                                           primary_key_fields, prompt, sample_size)
            
            # Generate data using LLM
            df = self._generate_data_with_llm(llm_schema, full_prompt, sample_size)
        else:
            # No LLM columns → start with empty rows
            df = pd.DataFrame([{}] * sample_size)
        
        # Apply type-based and column-specific generators
        df = self._apply_type_generators(df, llm_schema)
        
        # Convert column types based on schema
        df = self._convert_column_types(df, llm_schema)
        
        # Save output if path provided and return
        saved_path = self._save_output(df, output_path)
        if saved_path:
            return saved_path
            
        return df
