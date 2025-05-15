"""
Structured synthetic data generation using LLMs with SQLAlchemy integration.
"""

import pandas as pd
from pydantic import create_model, TypeAdapter
from typing import Dict, Optional, List, Union, Callable, Tuple, Type, Any
import os
import networkx as nx
import json
import openai

from .schemas import ModelConfig
from .llm import create_llm_client, LLMClient

try:
    from sqlalchemy.orm import DeclarativeMeta
    from sqlalchemy import inspect
except ImportError:
    DeclarativeMeta = None

def sqlalchemy_model_to_schema(model_class) -> Tuple[dict, dict, str]:
    """
    Convert a SQLAlchemy declarative model class to a schema dict, metadata dict,
    and extract docstrings.
    
    Returns:
        tuple: (schema_dict, metadata_dict, model_docstring)
            - schema_dict: Dictionary mapping column names to data types
            - metadata_dict: Dictionary containing column metadata (comments, constraints, etc.)
            - model_docstring: Docstring of the model class if available
    """
    schema = {}
    metadata = {}
    
    # Extract model docstring if available
    model_docstring = model_class.__doc__ or ""
    model_docstring = model_docstring.strip()
    
    # Process columns
    for col in model_class.__table__.columns:
        # Handle data type
        if col.foreign_keys:
            schema[col.name] = 'foreign_key'
        elif hasattr(col.type, 'python_type'):
            py_type = col.type.python_type
            
            # Map python types to our schema types
            if py_type == str:
                schema[col.name] = 'text'
            elif py_type in (int, float):
                schema[col.name] = 'number'
            elif py_type == bool:
                schema[col.name] = 'boolean'
            elif py_type.__name__ == 'date':
                schema[col.name] = 'date'
            elif py_type.__name__ == 'datetime':
                schema[col.name] = 'datetime'
            else:
                schema[col.name] = 'text'  # default to text for unknown types
        else:
            schema[col.name] = 'text'  # default type
            
        # Add column metadata
        col_metadata = {}
        
        # Extract column comment if available
        if col.comment:
            col_metadata['description'] = col.comment
            
        # Extract constraints
        constraints = []
        if col.primary_key:
            constraints.append('primary_key')
        if not col.nullable:
            constraints.append('not_null')
        if col.unique:
            constraints.append('unique')
            
        if constraints:
            col_metadata['constraints'] = constraints
            
        # Extract string length for varchar/text
        if hasattr(col.type, 'length') and col.type.length is not None:
            col_metadata['length'] = col.type.length
            
        # Extract default value
        if col.default is not None and not callable(col.default.arg):
            col_metadata['default'] = str(col.default.arg)
            
        # Add to metadata dict if we collected any metadata
        if col_metadata:
            metadata[col.name] = col_metadata
            
    return schema, metadata, model_docstring


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

    def generate_related_data(
        self,
        sqlalchemy_models: Union[List[Type], Type, str],
        prompts: Optional[Dict[str, str]] = None,
        sample_sizes: Optional[Dict[str, int]] = None,
        output_dir: Optional[str] = None,
        default_sample_size: int = 10,
        default_prompt: str = "Generate realistic data for this model.",
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
            
            results = generator.generate_related_data(
                sqlalchemy_models=[Customer, Order, OrderItem, Product],
                prompts={"Customer": "Generate tech companies"},
                sample_sizes={"Customer": 10, "Order": 30},
                custom_generators=custom_gens
            )
        """
        if prompts is None:
            prompts = {}
        if sample_sizes is None:
            sample_sizes = {}
            
        # Handle single SQLAlchemy model case
        if not isinstance(sqlalchemy_models, list):
            sqlalchemy_models = [sqlalchemy_models]
            
        # Create model dependency graph
        G = nx.DiGraph()
        
        # Store model information
        model_info = {}
        
        # Add nodes to the graph for each SQLAlchemy model
        for sqlalchemy_model in sqlalchemy_models:
            model_name = sqlalchemy_model.__name__
            G.add_node(model_name)
            model_info[model_name] = {
                'class': sqlalchemy_model,
                'foreign_keys': {},
                'references': set()
            }
            
        # Build the dependency graph based on foreign keys, not relationships
        for sqlalchemy_model in sqlalchemy_models:
            model_name = sqlalchemy_model.__name__
            
            # Get foreign key information directly from the table
            for column in sqlalchemy_model.__table__.columns:
                # Check if this column has any foreign keys
                if column.foreign_keys:
                    for fk in column.foreign_keys:
                        # Get the target table name and column
                        target_table = fk.column.table.name
                        target_column = fk.column.name
                        
                        # Find the SQLAlchemy model class that corresponds to this table
                        target_model_name = None
                        for m in sqlalchemy_models:
                            if m.__table__.name == target_table and m.__name__ in model_info:
                                target_model_name = m.__name__
                                break
                        
                        # Only process if we found a matching model in our list
                        if target_model_name:
                            # Add edge from target to source (dependency direction)
                            # This ensures we generate parents before children
                            G.add_edge(target_model_name, model_name)
                            
                            # Store foreign key mapping for later use
                            local_name = column.name
                            model_info[model_name]['foreign_keys'][local_name] = {
                                'target_model': target_model_name,
                                'target_column': target_column
                            }
                            
                            # Record that this model references the target
                            model_info[model_name]['references'].add(target_model_name)
            
        # Determine generation order - topological sort of dependency graph
        try:
            generation_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Handle cycles in dependency graph
            raise ValueError("Circular dependencies detected between SQLAlchemy models. Cannot determine generation order.")
            
        # Dictionary to hold generated data
        results = {}
        
        # Set up initial variables
        if custom_generators is None:
            custom_generators = {}
        
        # Store the original generators to restore them later
        original_generators = {
            'type': self.type_generators.copy(),
            'column': self.column_generators.copy()
        }
        
        try:
            # Generate data for each model in correct order
            for model_name in generation_order:
                model_class = model_info[model_name]['class']
                
                # Get sample size and prompt for this model
                sample_size = sample_sizes.get(model_name, default_sample_size)
                prompt = prompts.get(model_name, default_prompt)
                
                # Register any custom generators for this specific model
                if model_name in custom_generators:
                    for column_name, generator_func in custom_generators[model_name].items():
                        # Register this generator specifically for this column
                        self.register_generator('custom', generator_func, column_name=column_name)
                
                # Prepare foreign key generators if needed
                # Keep track of foreign key requirements for this model
                fk_requirements = {}
                for fk_col, fk_info in model_info[model_name]['foreign_keys'].items():
                    target_model = fk_info['target_model']
                    target_col = fk_info['target_column']
                    
                    # Only process for models we've already generated data for
                    if target_model in results:
                        # Store for later use in column generation
                        fk_requirements[fk_col] = {
                            'target_model': target_model,
                            'target_col': target_col,
                            'values': results[target_model][target_col].tolist() if target_col in results[target_model].columns else None
                        }
                
                # Ensure we have the schema for this model
                schema, meta_data, model_doc = sqlalchemy_model_to_schema(model_class)
                
                # Apply custom generators first if they exist for this model
                model_custom_gens = {}
                if model_name in custom_generators:
                    model_custom_gens = custom_generators[model_name]
                    print(f"Found {len(model_custom_gens)} custom generators for {model_name}")
                
                # Generate data for this model
                print(f"\nGenerating data for {model_name} with {len(schema)} columns")
                print(f"Schema: {schema}")
                print(f"Using prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Using prompt: {prompt}")
                
                # Generate the data
                df = self.generate_data(model_class, prompt, sample_size)
                
                # Ensure all expected columns are in the DataFrame
                for col_name in schema.keys():
                    if col_name not in df.columns:
                        print(f"⚠️ Adding missing column '{col_name}' to {model_name}")
                        # Check if this is a foreign key column first
                        if col_name in fk_requirements and fk_requirements[col_name]['values']:
                            fk_info = fk_requirements[col_name]
                            print(f"  Using foreign key values from {fk_info['target_model']}.{fk_info['target_col']} for {col_name}")
                            # Randomly select valid foreign key values
                            import random
                            valid_values = fk_info['values']
                            df[col_name] = [random.choice(valid_values) for _ in range(len(df))]
                        
                        # Use a custom generator if available
                        elif col_name in model_custom_gens:
                            print(f"  Using custom generator for {col_name}")
                            df[col_name] = [model_custom_gens[col_name](pd.Series(), col_name) for _ in range(len(df))]
                        
                        # Otherwise use an appropriate default based on column type
                        else:
                            col_type = schema[col_name]
                            if col_name == 'id':
                                # For primary IDs, use sequential numbers
                                df[col_name] = range(1, len(df) + 1)
                            elif col_name.endswith('_id') and col_name not in fk_requirements:
                                # For other ID columns that aren't foreign keys, use sequential numbers
                                df[col_name] = range(1, len(df) + 1)
                            elif col_type == 'text' or col_type == 'string':
                                df[col_name] = [f"{col_name}_{i}" for i in range(len(df))]
                            elif col_type == 'number':
                                df[col_name] = [i * 10 for i in range(len(df))]
                            elif col_type == 'boolean':
                                df[col_name] = [i % 2 == 0 for i in range(len(df))]
                            elif col_type == 'date' or col_type == 'datetime':
                                import datetime
                                base_date = datetime.date.today()
                                df[col_name] = [base_date + datetime.timedelta(days=i) for i in range(len(df))]
                            else:
                                df[col_name] = [f"default_{col_name}_{i}" for i in range(len(df))]
                
                # Apply any remaining custom generators that might not have been caught
                if model_name in custom_generators:
                    for col_name, gen_func in custom_generators[model_name].items():
                        if col_name in df.columns:
                            print(f"Applying custom generator for {model_name}.{col_name}")
                            df[col_name] = df.apply(lambda row: gen_func(row, col_name), axis=1)
                
                # Store the results
                results[model_name] = df
                
                # Clean up model-specific custom generators to avoid affecting the next model
                self.type_generators = original_generators['type'].copy()
                self.column_generators = original_generators['column'].copy()
                
                # Save to CSV if output_dir is provided
                if output_dir:
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{model_name.lower()}.csv")
                    df.to_csv(output_path, index=False)
                    
        except Exception as e:
            # Restore original generators in case of error
            self.type_generators = original_generators['type']
            self.column_generators = original_generators['column']
            raise e
            
        return results

    def generate_data(self, schema, prompt: str = "Generate synthetic data", sample_size: int = 10, 
                     output_path: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        Generate synthetic data based on schema using AI.
        
        Args:
            schema: Either a dictionary mapping field names to types,
                   a SQLAlchemy model class, or a path to a JSON schema file
            prompt: Description of what kind of data to generate
            sample_size: Number of records to generate
            output_path: Optional path to save generated data (csv or json)
            
        Returns:
            pandas DataFrame of generated data, or path to output file if output_path provided
        """
        # Determine schema type and convert to standard format
        llm_schema = {}
        metadata = {}
        model_description = None
        
        # Case 1: Dictionary schema
        if isinstance(schema, dict):
            llm_schema = schema
        
        # Case 2: SQLAlchemy model
        elif DeclarativeMeta is not None and isinstance(schema, type) and issubclass(schema, DeclarativeMeta):
            llm_schema, metadata, model_description = sqlalchemy_model_to_schema(schema)
        
        # Case 3: Path to JSON schema (not implemented yet)
        elif isinstance(schema, str) and (schema.endswith('.json') or schema.endswith('.schema')):
            with open(schema, 'r') as f:
                llm_schema = json.load(f)
        
        # Generate data using the LLM for fields without custom generators
        df = None
        if llm_schema:
            # Build Pydantic model for parsing
            fields = {col: (str, ...) for col in llm_schema}
            DynamicInstructorModel = create_model("DynamicModel", **fields)

            # Build prompt
            lines = [f"Generate {sample_size} records JSON objects with these fields:"]
            
            # Add model description if available
            if model_description:
                lines.append(f"Model Description: {model_description}")
                
            # Check if we have a primary key field and ensure it's explicitly noted
            primary_key_fields = []
            for col, col_meta in metadata.items():
                if 'constraints' in col_meta and 'primary_key' in col_meta['constraints']:
                    primary_key_fields.append(col)
            
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
            full_prompt = "\n".join(lines)

            print(f"Generating data using {self.model_config.provider}/{self.model_config.model_name}...")
            
            # Get model kwargs with proper API key handling
            model_kwargs = self.llm_client.get_model_kwargs()  # This ensures api_keys are not passed directly
            
            # Ensure the model name is always included
            if 'model' not in model_kwargs:
                model_kwargs['model'] = self.model_config.model_name
            
            try:
                # Call the LLM through instructor's unified interface
                ai_objs = self.client.chat.completions.create(
                    response_model=List[DynamicInstructorModel],
                    messages=[{"role": "user", "content": full_prompt}],
                    **model_kwargs,
                )
                
                if not ai_objs:
                    print(f"⚠️ Warning: No objects returned from LLM call")
                    df = pd.DataFrame([{col: f"dummy_{i}_{col}" for col in llm_schema.keys()} for i in range(sample_size)])
                else:
                    # Convert objects to DataFrame
                    records = [obj.model_dump() for obj in ai_objs]
                    
                    # Debug log
                    if not records:
                        print(f"⚠️ Warning: No records extracted from LLM response")
                    else:
                        print(f"✓ Successfully generated {len(records)} records")
                        print(f"  Fields in first record: {list(records[0].keys()) if records else 'None'}")
                    
                    # Create DataFrame from records
                    df = pd.DataFrame(records)
                    
                    # Ensure all expected columns are present
                    for col in llm_schema.keys():
                        if col not in df.columns:
                            print(f"⚠️ Missing column '{col}' in generated data. Adding default values.")
                            df[col] = [f"default_{col}_{i}" for i in range(len(df))] if len(df) > 0 else [f"default_{col}_0" for i in range(sample_size)]
                    
                    # If no data was returned, create dummy data
                    if df.empty:
                        print(f"⚠️ Warning: Empty DataFrame returned. Creating dummy data.")
                        df = pd.DataFrame([{col: f"dummy_{i}_{col}" for col in llm_schema.keys()} for i in range(sample_size)])
                
            except Exception as e:
                print(f"⚠️ Error generating data with instructor: {str(e)}")
                print("Creating fallback dummy data to continue execution")
                
                # Create a dummy DataFrame with the expected columns so processing can continue
                df = pd.DataFrame([{col: f"fallback_{i}_{col}" for col in llm_schema.keys()} for i in range(sample_size)])
                
            # Validate DataFrame structure
            print(f"DataFrame shape: {df.shape} with columns: {list(df.columns)}")
            
            # If we still have empty data after all efforts, make one last attempt
            if df.empty or len(df.columns) == 0:
                print("❌ Failed to generate valid data after multiple attempts. Creating backup data.")
                df = pd.DataFrame([{col: f"backup_{i}_{col}" for col in llm_schema.keys()} for i in range(sample_size)])
        else:
            # No LLM columns → start with empty rows
            df = pd.DataFrame([{}] * sample_size)
            
        # Apply custom generators for specific types and columns
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
                
        # Attempt to convert columns to appropriate types
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
        
        # Save output if path provided
        if output_path:
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
                return output_path
            elif output_path.endswith('.json'):
                df.to_json(output_path, orient='records')
                return output_path
                
        return df
