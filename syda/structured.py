import openai
import pandas as pd
from pydantic import create_model
from typing import Dict, Optional, List, Union, Callable, Tuple, Type
import instructor
from anthropic import Anthropic
import os
import networkx as nx

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
            if py_type is int:
                dtype = 'number'
            elif py_type is float:
                dtype = 'number'
            elif py_type is str:
                dtype = 'text'
            else:
                dtype = 'text'
            schema[col.name] = dtype
        else:
            schema[col.name] = 'text'
        
        # Collect metadata for this column
        col_metadata = {}
        
        # Extract SQLAlchemy comment if available
        if col.comment:
            col_metadata['description'] = col.comment
            
        # Extract constraints
        constraints = []
        if col.primary_key:
            constraints.append('primary_key')
        if col.unique:
            constraints.append('unique')
        if not col.nullable:
            constraints.append('not_null')
        if constraints:
            col_metadata['constraints'] = constraints
            
        # Extract default value if present
        if col.default is not None and not callable(getattr(col.default, 'arg', None)):
            col_metadata['default'] = str(col.default.arg)
        
        # Store length for string types
        if hasattr(col.type, 'length') and col.type.length is not None:
            col_metadata['length'] = col.type.length
            
        # If we found any metadata, store it
        if col_metadata:
            metadata[col.name] = col_metadata
    
    return schema, metadata, model_docstring


class SyntheticDataGenerator:
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        """
        Initialize the synthetic data generator.

        Args:
            model: Model to use ("gpt-4", "gpt-3.5-turbo", "claude-2", etc.).
            temperature: Sampling temperature.
        """
        self.model = model.lower()
        self.temperature = temperature

        # Pick and wrap the correct client once
        if self.model.startswith("claude"):
            raw_client = Anthropic()
        else:
            raw_client = openai.OpenAI()
        self.client = instructor.patch(raw_client)

        # Registry for custom generators by type: type_name -> fn(row: pd.Series, col_name: str) -> value
        self.type_generators: Dict[str, Callable[[pd.Series, str], any]] = {}
        
        # Registry for custom generators by column name: col_name -> fn(row: pd.Series, col_name: str) -> value
        self.column_generators: Dict[str, Callable[[pd.Series, str], any]] = {}

    def register_generator(self, type_name: str, func: Callable[[pd.Series, str], any], column_name: Optional[str] = None):
        """
        Register a custom generator for a specific data type or column name.

        Args:
            type_name: The name of the data type (e.g., 'icd10_code', 'foreign_key').
            func: A function that takes (row: pd.Series, col_name: str) and returns a generated value.
            column_name: Optional specific column name to register this generator for. If provided,
                       this generator will only be used for this specific column. This takes precedence
                       over type-based generators.
        """
        if column_name:
            # Register a column-specific generator
            self.column_generators[column_name] = func
        else:
            # Register a type-based generator
            self.type_generators[type_name.lower()] = func

    def generate_related_data(
        self,
        models: Union[List[Type], Type, str],
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
        1. Analyzes model dependencies (which models reference others)
        2. Automatically determines the correct order to generate data
        3. Handles foreign key relationships between models
        4. Applies custom generators where registered
        
        Args:
            models: A list of SQLAlchemy model classes, a single model class, 
                   or a string pattern to match class names
            prompts: Optional dictionary mapping model names to custom prompts
            sample_sizes: Optional dictionary mapping model names to sample sizes
            output_dir: Optional directory to save CSV files (one per model)
            default_sample_size: Default number of records if not specified in sample_sizes
            default_prompt: Default prompt if not specified in prompts
            custom_generators: Optional dictionary specifying custom generators for models and columns.
                              Format: {"ModelName": {"column_name": generator_function}}
                              where generator_function is a callable that takes (row: pd.Series, col_name: str)
                              and returns a generated value.
            
        Returns:
            Dictionary mapping model names to DataFrames of generated data
            
        Example:
            # Define custom generators for specific columns in specific models
            custom_gens = {
                "Customer": {
                    "status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"])
                },
                "Product": {
                    "price": lambda row, col: round(random.uniform(50, 500), 2)
                }
            }
            
            results = generator.generate_related_data(
                models=[Customer, Order, OrderItem, Product],
                prompts={"Customer": "Generate tech companies"},
                sample_sizes={"Customer": 10, "Order": 30},
                custom_generators=custom_gens
            )
        """
        if prompts is None:
            prompts = {}
        if sample_sizes is None:
            sample_sizes = {}
            
        # Handle single model case
        if not isinstance(models, list):
            models = [models]
            
        # Create model dependency graph
        G = nx.DiGraph()
        
        # Store model information
        model_info = {}
        
        # Add nodes to the graph for each model
        for model in models:
            model_name = model.__name__
            G.add_node(model_name)
            model_info[model_name] = {
                'class': model,
                'foreign_keys': {},
                'references': set()
            }
            
        # Build the dependency graph based on foreign keys, not relationships
        for model in models:
            model_name = model.__name__
            
            # Get foreign key information directly from the table
            for column in model.__table__.columns:
                # Check if this column has any foreign keys
                if column.foreign_keys:
                    for fk in column.foreign_keys:
                        # Get the target table name and column
                        target_table = fk.column.table.name
                        target_column = fk.column.name
                        
                        # Find the model class that corresponds to this table
                        target_model_name = None
                        for m in models:
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
            raise ValueError("Circular dependencies detected between models. Cannot determine generation order.")
            
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
                for fk_col, fk_info in model_info[model_name]['foreign_keys'].items():
                    target_model = fk_info['target_model']
                    target_col = fk_info['target_column']
                    
                    # Only set up generators for models we've already processed
                    if target_model in results:
                        # Get valid values from the target model's generated data
                        valid_values = results[target_model][target_col].tolist()
                        
                        if valid_values:
                            # Register a generator that will randomly select from valid IDs
                            def make_fk_generator(values):
                                # Need to use factory function to avoid closure issues
                                return lambda row, col: pd.Series(values).sample(1).iloc[0]
                            
                            # Register this generator specifically for this column
                            # Only if a custom generator hasn't been specified for this column
                            if model_name not in custom_generators or fk_col not in custom_generators[model_name]:
                                self.register_generator('foreign_key', make_fk_generator(valid_values), column_name=fk_col)
                
                # Generate data for this model
                df = self.generate_data(model_class, prompt, sample_size)
                
                # Store the results
                results[model_name] = df
                
                # Clean up model-specific custom generators to avoid affecting the next model
                if model_name in custom_generators:
                    for column_name in custom_generators[model_name].keys():
                        if column_name in self.column_generators:
                            # Only remove generators that were added by this method
                            if column_name not in original_generators['column']:
                                del self.column_generators[column_name]
        finally:
            # Restore the original generators to avoid side effects
            # This ensures any temporary generators we added don't persist beyond this method call
            self.type_generators = original_generators['type']
            self.column_generators = original_generators['column']
            
            # Save to file if output directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{model_name}.csv")
                df.to_csv(output_path, index=False)
                
        return results
        
    def generate_data(
        self,
        schema: Union[Dict[str, str], type],
        prompt: str,
        sample_size: int = 10,
        output_path: Optional[str] = None
    ) -> Union[pd.DataFrame, str]:
        """
        Generate synthetic data via Instructor‑patched LLM, then apply custom generators.

        Args:
            schema: Either a mapping of column names to data types (all treated as strings),
                or a SQLAlchemy declarative model class. If a model class is provided, columns
                with foreign keys will be assigned the 'foreign_key' type.
            prompt: Description of the data to generate.
            sample_size: Number of records to generate.
            output_path: Optional CSV or JSON filepath to save results.

        Usage:
            # Register a generator for foreign keys
            generator.register_generator('foreign_key', lambda row, col: random.choice([1, 2, 3]))
            # Pass a SQLAlchemy model class directly
            generator.generate_data(MyModel, prompt, sample_size=10)

        Returns:
            DataFrame of generated records, or filepath if output_path is set.
        """
        # If schema is a SQLAlchemy model class, extract schema dict, metadata, and docstring
        if DeclarativeMeta is not None and isinstance(schema, DeclarativeMeta):
            schema, metadata, model_docstring = sqlalchemy_model_to_schema(schema)
        else:
            metadata = {}
            model_docstring = ""

        # 1) Identify which cols the LLM should handle vs. custom-only
        custom_cols = set()
        for col, dtype in schema.items():
            # Check if we have a column-specific generator
            if col in self.column_generators:
                custom_cols.add(col)
            # Check if we have a type-based generator
            elif dtype.lower() in self.type_generators:
                custom_cols.add(col)
        
        # Columns to be handled by LLM are those not handled by custom generators
        llm_schema = {
            col: dtype for col, dtype in schema.items()
            if col not in custom_cols
        }

        # 2) Generate LLM-driven columns (if any)
        if llm_schema:
            # Build Pydantic model for parsing
            fields = {col: (str, ...) for col in llm_schema}
            DynamicModel = create_model("DynamicModel", **fields)

            # Build prompt
            lines = [f"Generate {sample_size} records JSON objects with these fields:"]
            
            # Include model description if available
            if model_docstring:
                lines.insert(0, f"Model Description: {model_docstring}")
                
            # Include field descriptions and constraints
            for col, dtype in llm_schema.items():
                field_desc = f"- {col}: {dtype}"
                
                # Add field metadata if available
                if col in metadata:
                    col_meta = metadata[col]
                    
                    # Add description
                    if 'description' in col_meta:
                        field_desc += f" - {col_meta['description']}"
                    
                    # Add constraints
                    if 'constraints' in col_meta:
                        constraints = ", ".join(col_meta['constraints'])
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

            # Call the LLM
            ai_objs: List[DynamicModel] = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                response_model=List[DynamicModel],
                temperature=self.temperature,
            )
            df = pd.DataFrame([obj.dict() for obj in ai_objs])
        else:
            # No LLM columns → start with empty rows
            df = pd.DataFrame([{}] * sample_size)

        # 3) Populate custom columns using the full row context
        for col in custom_cols:
            # First check if we have a column-specific generator
            if col in self.column_generators:
                gen_fn = self.column_generators[col]
            # Otherwise use the type-based generator
            else:
                gen_fn = self.type_generators[schema[col].lower()]
            
            # Apply the generator to create values for this column
            df[col] = df.apply(lambda row: gen_fn(row, col), axis=1)

        # 4) Optionally save to CSV or JSON
        if output_path:
            p = output_path.lower()
            if p.endswith(".csv"):
                df.to_csv(output_path, index=False)
            elif p.endswith(".json"):
                df.to_json(output_path, orient="records")
            else:
                raise ValueError("output_path must end with .csv or .json")
            return output_path

        return df
