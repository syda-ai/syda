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
from .dependency_handler import DependencyHandler, ForeignKeyHandler
from .custom_generators import GeneratorManager
from .schema_loader import SchemaLoader

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
        
        # Initialize the generator manager
        self.generator_manager = GeneratorManager()
        
        # Initialize the foreign key handler
        self.fk_handler = ForeignKeyHandler(self.generator_manager)
        
        # Initialize the schema loader
        self.schema_loader = SchemaLoader()
        
        # For backward compatibility, provide direct access to these dictionaries
        self.type_generators = self.generator_manager.type_generators
        self.column_generators = self.generator_manager.column_generators

    def register_generator(self, type_name: str, func: Callable[[pd.Series, str], any], column_name: Optional[str] = None):
        """
        Register a custom generator for a specific data type or column name.
        
        Args:
            type_name: The data type this generator handles (e.g., 'number', 'text', 'foreign_key')
            func: Function that takes (row: pd.Series, col_name: str) and returns a generated value
            column_name: If specified, this generator only applies to the named column
                        rather than all columns of the specified type
        """
        # Delegate to the generator manager
        self.generator_manager.register_generator(type_name.lower(), func, column_name)

    def _sqlalchemy_models_to_schemas(self, sqlalchemy_models):
        """
        Convert a list of SQLAlchemy models to the schema format expected by generate_for_schemas.
        
        Args:
            sqlalchemy_models: List of SQLAlchemy model classes
            
        Returns:
            Dictionary mapping schema names to schema definitions
        """
        schemas = {}
        
        for model_class in sqlalchemy_models:
            # Use our SchemaLoader to load the model
            schema_dict = self.schema_loader.load_schema(model_class)
            
            # Determine the table name (either from tablename attribute or class name)  
            schema_name = model_class.__tablename__
            schemas[schema_name] = schema_dict
        # print("schemas", schemas)
        # exit(0)
        return schemas
    
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
        
    def _process_foreign_keys(self, schema_foreign_keys):
        """
        Process foreign keys extracted from schemas into a standardized format.
        
        Args:
            schema_foreign_keys: Dictionary mapping schema names to foreign key definitions
            
        Returns:
            Dictionary of standardized foreign key definitions
        """
        extracted_foreign_keys = {}
        
        # Process the foreign keys extracted from schema files
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
                
        return extracted_foreign_keys

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
        template_schemas = {}
        schema_depends_on_schemas = {}
        # Dictionary to store extracted foreign keys
        schema_foreign_keys = {}
        
        for schema_name, schema_source in schemas.items():
            llm_schema, metadata, desc, extracted_fks, template_fields, depends_on_schemas = self.schema_loader.load_schema(schema_source)
            # Store processed schema and metadata
            processed_schemas[schema_name] = llm_schema
            schema_depends_on_schemas[schema_name] = depends_on_schemas
            schema_metadata[schema_name] = metadata or {}
            schema_descriptions[schema_name] = desc or f"{schema_name} data"
            if "__template__" in template_fields:
                template_schemas[schema_name] = template_fields
            # Store extracted foreign keys if any
            if extracted_fks:
                schema_foreign_keys[schema_name] = extracted_fks
        # Extract and process foreign keys from schemas
        extracted_foreign_keys = self._process_foreign_keys(schema_foreign_keys)
        
        # Use DependencyHandler to extract all dependencies
        all_dependencies = DependencyHandler.extract_dependencies(
            schemas=schemas,
            schema_metadata=schema_metadata,
            foreign_keys=extracted_foreign_keys,
            schema_depends_on_schemas=schema_depends_on_schemas
        )
        
        # Calculate the generation order based on all dependencies using DependencyHandler
        generation_order = list(schemas.keys())
        
        try:
            # Build dependency graph and determine generation order
            dependency_graph = DependencyHandler.build_dependency_graph(
                nodes=list(schemas.keys()),
                dependencies=all_dependencies
            )
            generation_order = DependencyHandler.determine_generation_order(dependency_graph)
            
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
        
        try:
            # Generate structured data using the extracted method
            results = self._generate_structured_data(
                processed_schemas=processed_schemas,
                schema_metadata=schema_metadata,
                schema_descriptions=schema_descriptions,
                generation_order=generation_order,
                extracted_foreign_keys=extracted_foreign_keys,
                prompts=prompts,
                sample_sizes=sample_sizes,
                custom_generators=custom_generators,
                default_prompt=default_prompt,
                default_sample_size=default_sample_size
            )
          
            # Separate template schemas from structured schemas
            template_schemas_dfs = {}
            structured_results = {}
            #print("results: ", results)
            for schema_name, df in results.items():
                # Check if this is a template schema by looking for __template_source__ field
                if df is not None and schema_name in template_schemas:
                    template_schemas_dfs[schema_name] = (df, template_schemas[schema_name])
                else:
                    structured_results[schema_name] = df
            
            # Process template schemas if any
            template_results = {}
            print("\n===== Processing TEMPLATE SCHEMAS =====", template_schemas_dfs.keys())
            if template_schemas_dfs and output_dir:
                # Process each template schema using the TemplateProcessor
                from syda.templates import TemplateProcessor
                processor = TemplateProcessor()
                
                # Use the new method to process all template dataframes at once
                template_results = processor.process_template_dataframes(template_schemas_dfs, output_dir)
            
            # Save files if output_dir is specified
            if output_dir:
                save_dataframes(structured_results, output_dir, format=output_format)
            
            # Verify referential integrity using ForeignKeyHandler
            self.fk_handler.verify_referential_integrity(results, extracted_foreign_keys)
                    
        except Exception as e:
            raise e
            
        return results


    def _generate_structured_data(
        self, 
        processed_schemas,
        schema_metadata,
        schema_descriptions,
        generation_order, 
        extracted_foreign_keys,
        prompts, sample_sizes,
        custom_generators,
        default_prompt, 
        default_sample_size
    ):
        """
        Generate structured data for each schema in the specified generation order.
        
        Args:
            processed_schemas: Dictionary of processed schemas
            schema_metadata: Dictionary of metadata for each schema
            schema_descriptions: Dictionary of descriptions for each schema
            generation_order: List of schema names in the order they should be generated
            extracted_foreign_keys: Dictionary of foreign key relationships
            prompts: Dictionary of prompts for each schema
            sample_sizes: Dictionary of sample sizes for each schema
            custom_generators: Dictionary of custom generators for each schema
            default_prompt: Default prompt to use if no schema-specific prompt is provided
            default_sample_size: Default sample size to use if no schema-specific sample size is provided
            
        Returns:
            Dictionary mapping schema names to generated DataFrames
        """
        results = {}
        
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
            
            # Apply foreign key constraints using the ForeignKeyHandler
            self.fk_handler.apply_foreign_keys(schema_name, extracted_foreign_keys, results)
            
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
                # Pass the already extracted schema information to avoid redundant extraction
                df = self._generate_data(
                    table_schema=llm_schema, 
                    metadata=metadata, 
                    table_description=model_description,
                    prompt=prompt, 
                    sample_size=sample_size,
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
            # Apply custom generators if any
            schema_custom_generators = custom_generators.get(schema_name, {})
            print(f"Applying custom generators for schema {schema_name}")
            df = self.generator_manager.apply_custom_generators(
                df, schema_name, schema_custom_generators, parent_dfs=results)
            # Store the result
            results[schema_name] = df
            
        return results
        
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
            
            # Create DataFrame from records
            df = pd.DataFrame(records)
            
            # Ensure all expected columns are present
            for col in llm_schema.keys():
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in generated data. Data generation failed to produce the expected schema.")
            
            # If no data was returned, fail
            if df.empty:
                raise ValueError("Empty DataFrame returned from data generation")
            return df
                
        except Exception as e:
            raise ValueError(f"Error generating data: {str(e)}")

    def _apply_type_generators(self, df, llm_schema):
        """
        Apply custom type-based and column-specific generators to the data.
        
        Args:
            df: DataFrame to apply generators to
            llm_schema: Dictionary mapping field names to types
            
        Returns:
            DataFrame with generators applied
        """
        # Delegate to the generator manager
        return self.generator_manager.apply_type_generators(df, llm_schema)

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
                dtype = llm_schema[col].lower() if isinstance(llm_schema[col], str) else llm_schema[col].get('type', '').lower()
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
            
        # Apply column-specific generators
        for col_name in df.columns:
            if col_name in self.column_generators:
                print(f"Applying custom generator for {col_name}")
                df[col_name] = df.apply(lambda row: self.column_generators[col_name](row, col_name), axis=1)
        
        return df
