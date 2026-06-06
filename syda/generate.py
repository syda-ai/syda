"""
Structured synthetic data generation using LLMs with SQLAlchemy integration.

This module provides the primary interface for generating synthetic structured data
using Large Language Models (LLMs) with the option to use SQLAlchemy models.

The SyntheticDataGenerator class is the main entry point for generating synthetic
structured data. It takes in a set of schemas (either as YAML/JSON files or Python
dicts) and a set of prompts to generate data for those schemas. The generator
supports generating data for multiple related schemas, including automatically
resolving foreign key dependencies.

The module also provides support for generating unstructured data (such as text or
PDF documents) using templates and LLMs. See the `syda.unstructured` module for
more information.

The library is designed to be extensible, with the ability to add new LLM providers
and custom generators for specific data types.
"""

import pandas as pd
import json
import os
import math
import time
import random
import pkgutil
import importlib
import inspect
from pathlib import Path
import networkx as nx
from typing import Dict, List, Optional, Callable, Union, Type, Any, Tuple
from pydantic import create_model, TypeAdapter, Field
from .schemas import ModelConfig
from .llm import create_llm_client, LLMClient
from .output import save_dataframe, save_dataframes, append_dataframe

_STREAM_THRESHOLD = 10_000  # rows; above this, chunks are written directly to disk when output_dir is set


def _stream_fmt(path: str) -> str:
    """Infer 'json' or 'csv' from a file path for append_dataframe."""
    return "json" if path.endswith(".json") else "csv"


def _slim(df: "pd.DataFrame", fk_cols: Optional[set], schema: dict) -> "pd.DataFrame":
    """Return df reduced to fk_cols, or an empty-schema frame if none needed."""
    if fk_cols:
        keep = list(fk_cols & set(df.columns))
        if keep:
            return df[keep]
    return pd.DataFrame(columns=list(schema.keys()))
from .utils import (
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
    
    def __init__(
        self,
        model_config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        grok_api_key: Optional[str] = None
    ):
        """
        Initialize the synthetic data generator with the specified model configuration.
        
        Args:
            model_config: Configuration for the AI model to use, either as a ModelConfig object 
                         or a dictionary of parameters. If None, default settings will be used.
            openai_api_key: Optional API key for OpenAI. If not provided, will use OPENAI_API_KEY 
                           environment variable.
            anthropic_api_key: Optional API key for Anthropic. If not provided, will use 
                              ANTHROPIC_API_KEY environment variable.
            gemini_api_key: Optional API key for Gemini. If not provided, will use GEMINI_API_KEY
            grok_api_key: Optional API key for Grok. If not provided, will use GROK_API_KEY
        """
        # Initialize the LLM client using our new module
        self.llm_client = create_llm_client(
            model_config=model_config,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            gemini_api_key=gemini_api_key,
            grok_api_key=grok_api_key
        )
        
        # Store the model configuration for easy access
        self.model_config = self.llm_client.model_config
        
        # Initialize the generator manager
        self.generator_manager = GeneratorManager()
        
        # Initialize the foreign key handler
        self.fk_handler = ForeignKeyHandler(self.generator_manager)
        
        # Initialize the schema loader
        self.schema_loader = SchemaLoader()
        
        # For backward compatibility, provide direct access to these dictionaries
        self.type_generators = self.generator_manager.type_generators
        self.column_generators = self.generator_manager.column_generators

    def _sqlalchemy_models_to_schemas(
        self,
        sqlalchemy_models: Union[List[Type], Type, str],
    ):
        """
        Convert a list of SQLAlchemy models to the schema format expected by generate_for_schemas.
        
        This function takes a list of SQLAlchemy model classes and converts them to the dictionary
        format expected by generate_for_schemas. The output is a dictionary mapping schema names
        (i.e. table names) to schema definitions, which are dictionaries containing the field
        names and their metadata.
        
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
        output_format: str = 'csv',
        batch_size: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for multiple relational SQLAlchemy models with automatic 
        dependency resolution based on foreign key relationships.
        
        This function supports both regular SQLAlchemy models and models with template attributes
        by converting them to schema format and then using the generate_for_schemas method for data generation.
        
        This function:
        1. Converts SQLAlchemy models to schema format
        2. Calls generate_for_schemas to handle the data generation
        
        Args:
            sqlalchemy_models: A list of SQLAlchemy model classes, a single SQLAlchemy model class, 
                    or a string pattern to match class names
            prompts: Optional dictionary mapping table names to custom prompts
            sample_sizes: Optional dictionary mapping table names to sample sizes
            output_dir: Optional directory to save files (one per table)
            default_sample_size: Default number of records if not specified in sample_sizes
            default_prompt: Default prompt if not specified in prompts
            custom_generators: Optional dictionary specifying custom generators for SQLAlchemy models and columns.
                              Format: {"TableName": {"column_name": generator_function}}
            output_format: Format to use when saving files ('csv' or 'json')
            
        Returns:
            Dictionary mapping table names to DataFrames of generated data
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
            output_format=output_format,
            batch_size=batch_size,
        )
        
    def _process_foreign_keys(
        self, 
        schema_foreign_keys: Dict[str, Dict[str, Tuple[str, str]]],
    ):
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
        output_format: str = 'csv',
        batch_size: Optional[int] = None,
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
            
            print("\n[INFO] Generation order determined:")
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
            results, streamed_schemas = self._generate_structured_data(
                processed_schemas=processed_schemas,
                schema_metadata=schema_metadata,
                schema_descriptions=schema_descriptions,
                generation_order=generation_order,
                extracted_foreign_keys=extracted_foreign_keys,
                prompts=prompts,
                sample_sizes=sample_sizes,
                custom_generators=custom_generators,
                default_prompt=default_prompt,
                default_sample_size=default_sample_size,
                batch_size=batch_size,
                output_dir=output_dir,
                output_format=output_format,
                template_schema_names=set(template_schemas.keys()),
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
            
            # Save files if output_dir is specified (skip tables already streamed to disk)
            if output_dir:
                to_save = {k: v for k, v in structured_results.items()
                           if k not in streamed_schemas}
                if to_save:
                    save_dataframes(to_save, output_dir, format=output_format)
            
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
        default_sample_size,
        batch_size: Optional[int] = None,
        output_dir: Optional[str] = None,
        output_format: str = 'csv',
        template_schema_names: Optional[set] = None,
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
            Tuple of (results_dict, streamed_schemas_set).
            streamed_schemas_set contains names of tables already written to output_dir.
            When output_dir is set, results entries are slimmed to FK-columns only after
            being written — callers should read full data from disk.
        """
        _template_names = template_schema_names or set()
        ext = output_format if output_format in ('csv', 'json') else 'csv'

        # Pre-compute which columns each table must expose to FK children.
        fk_exposed: Dict[str, set] = {}
        for fk_defs in extracted_foreign_keys.values():
            for _child_col, (parent_schema, parent_col) in fk_defs.items():
                fk_exposed.setdefault(parent_schema, set()).add(parent_col)

        # Pre-compute the LAST generation-order index at which each parent is needed.
        # After that index is processed we can drop the parent's non-FK data from RAM.
        order_idx: Dict[str, int] = {t: i for i, t in enumerate(generation_order)}
        last_consumer: Dict[str, int] = {}  # parent -> last child index
        for child_tbl in generation_order:
            for _, (parent, _) in extracted_foreign_keys.get(child_tbl, {}).items():
                ci = order_idx[child_tbl]
                if ci > last_consumer.get(parent, -1):
                    last_consumer[parent] = ci

        results: Dict[str, Any] = {}
        streamed_schemas: set = set()

        for schema_name in generation_order:
            llm_schema = processed_schemas[schema_name]
            metadata = schema_metadata[schema_name]
            model_description = schema_descriptions[schema_name]

            print(f"\nGenerating data for {schema_name} with {len(llm_schema)} columns")
            print(f"Description: {model_description}")

            prompt = prompts.get(schema_name, default_prompt)
            sample_size = sample_sizes.get(schema_name, default_sample_size)

            self.fk_handler.apply_foreign_keys(schema_name, extracted_foreign_keys, results)

            print(f"Creating data for {schema_name} with schema: {llm_schema}")

            # Decide whether to stream chunks directly to disk during generation
            # (activates for large tables in direct mode to cap per-chunk RAM).
            _eff_mode = self.model_config.generation_mode
            if _eff_mode == 'auto':
                _eff_mode = 'direct' if sample_size <= 500 else 'codegen'
            stream_path: Optional[str] = None
            if output_dir and sample_size > _STREAM_THRESHOLD and _eff_mode == 'direct':
                stream_path = os.path.join(output_dir, f"{schema_name.lower()}.{ext}")
                os.makedirs(output_dir, exist_ok=True)
                print(f"[syda] Streaming {sample_size:,} rows → {stream_path}")

            try:
                df = self._generate_data(
                    table_schema=llm_schema,
                    metadata=metadata,
                    table_description=model_description,
                    prompt=prompt,
                    sample_size=sample_size,
                    batch_size=batch_size,
                    schema_name=schema_name,
                    stream_path=stream_path,
                    fk_cols_to_keep=fk_exposed.get(schema_name),
                )

                if stream_path and os.path.exists(stream_path):
                    streamed_schemas.add(schema_name)
                    print(f"[syda] {schema_name}: streamed {sample_size:,} rows to {stream_path}")
                elif len(df) < sample_size:
                    print(
                        f"Warning: LLM generated only {len(df)} records "
                        f"instead of {sample_size} for {schema_name}"
                    )
                elif len(df) > sample_size:
                    df = df.iloc[:sample_size]

            except Exception as e:
                print(f"Error using AI generation for {schema_name}: {str(e)}")
                raise Exception(f"Failed to generate data for {schema_name} using LLM: {str(e)}")

            schema_custom_generators = custom_generators.get(schema_name, {})
            print(f"Applying custom generators for schema {schema_name}")
            df = self.generator_manager.apply_custom_generators(
                df, schema_name, schema_custom_generators, parent_dfs=results)

            # ── Progressive disk write + memory slim ──────────────────────────
            # When output_dir is set, write every table to disk as soon as it's
            # ready (not just > 10K-row tables).  After writing, replace the
            # in-memory entry with FK-columns only so subsequent tables' FK
            # generators still work but full row data is freed.
            # Template tables keep their full DataFrames because the template
            # processor needs them later in this call.
            if output_dir and schema_name not in streamed_schemas and schema_name not in _template_names:
                file_path = os.path.join(output_dir, f"{schema_name.lower()}.{ext}")
                os.makedirs(output_dir, exist_ok=True)
                save_dataframe(df, file_path)
                streamed_schemas.add(schema_name)
                print(f"[syda] {schema_name}: {len(df):,} rows written to {file_path}")

                # Slim to FK columns; fully drop if nobody references this table.
                fk_cols = list(fk_exposed.get(schema_name, set()) & set(df.columns))
                df = df[fk_cols] if fk_cols else pd.DataFrame(columns=list(df.columns)[:0])

            results[schema_name] = df

            # ── Free parent tables fully consumed by this step ────────────────
            # A parent table is no longer needed once all its FK children have
            # been generated.  When output_dir is set (data is on disk), drop
            # its results entry entirely; otherwise keep FK columns for the
            # integrity check that runs at the end of generate_for_schemas().
            current_idx = order_idx[schema_name]
            for parent, last_idx in last_consumer.items():
                if last_idx == current_idx and parent in results:
                    if output_dir:
                        del results[parent]
                        print(f"[syda] Freed '{parent}' from memory (all FK children generated)")
                    else:
                        # No disk — keep only the FK columns for integrity check
                        fk_cols = list(fk_exposed.get(parent, set()) & set(results[parent].columns))
                        results[parent] = (
                            results[parent][fk_cols] if fk_cols
                            else pd.DataFrame(columns=list(results[parent].columns)[:0])
                        )

        return results, streamed_schemas
        
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
        def get_python_type(field_type):
            if isinstance(field_type, str):
                type_map = {
                    'integer': int, 'int': int, 'float': float, 'number': float,
                    'boolean': bool, 'bool': bool, 'array': list, 'object': dict,
                }
                return type_map.get(field_type.lower(), str)
            elif isinstance(field_type, dict) and 'type' in field_type:
                return get_python_type(field_type['type'])
            return str

        model_fields = {}
        for name, field_info in llm_schema.items():
            if isinstance(field_info, dict) and 'type' in field_info:
                model_fields[name] = (get_python_type(field_info['type']), Field(description=name))
            elif isinstance(field_info, str):
                model_fields[name] = (get_python_type(field_info), Field(description=name))
            else:
                model_fields[name] = (str, Field(description=name))

        DynamicModel = create_model("DynamicModel", **model_fields)

        try:
            print(f"Generating data using {self.model_config.provider}/{self.model_config.model_name}...")

            agent = self.llm_client.create_agent(
                output_type=List[DynamicModel],
                system_prompt=(
                    "You are a synthetic data generator. Generate realistic, diverse synthetic data "
                    "that matches the schema exactly. Return exactly the requested number of records."
                ),
            )
            model_settings = self.llm_client.get_model_settings()
            result = agent.run_sync(full_prompt, model_settings=model_settings)
            ai_objs = result.output

            if not ai_objs:
                raise ValueError(
                    f"No objects returned from LLM. "
                    f"Provider: {self.model_config.provider}/{self.model_config.model_name}. "
                    f"Check token limits and API connectivity. Sample size: {sample_size}."
                )

            records = [obj.model_dump() for obj in ai_objs]
            if not records:
                raise ValueError("No records extracted from LLM response")

            print(f"[OK] Successfully generated {len(records)} records")
            df = pd.DataFrame(records)

            for col in llm_schema.keys():
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in generated data.")

            if df.empty:
                raise ValueError("Empty DataFrame returned from data generation")
            return df

        except Exception as e:
            raise ValueError(f"Error generating data: {str(e)}")

    def _call_with_retry(self, fn, max_retries: int, base_delay: float = 1.0):
        """Call fn with exponential-backoff retry on transient errors."""
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                name = type(exc).__name__
                status = getattr(exc, 'status_code', None)
                if status in (401, 422) or 'Auth' in name or isinstance(exc, (ValueError, TypeError)):
                    raise
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    print(
                        f"[syda] Chunk failed ({name}), retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
        raise RuntimeError(f"Chunk failed after {max_retries} retries: {last_exc}") from last_exc

    def _resolve_batch_size(self, sample_size: int, batch_size: Optional[int] = None) -> int:
        """Auto-select batch size when not explicitly specified."""
        explicit = batch_size if batch_size is not None else self.model_config.batch_size
        if explicit is not None:
            return explicit
        if sample_size <= 50:
            return sample_size
        if sample_size <= 200:
            return 50
        if sample_size <= 1000:
            return 100
        return 200

    def _analyze_and_generate_code(
        self, table_schema: Dict, metadata: Dict, table_description: Optional[str], schema_name: str
    ) -> List[str]:
        """
        LLM analysis call for code-gen mode: classifies columns as simple or semantic,
        generates Python functions for simple columns, returns list of semantic column names.
        """
        from pydantic import BaseModel as _PydanticBase
        try:
            from pydantic_ai import ModelRetry
        except ImportError:
            ModelRetry = None

        class SchemaAnalysis(_PydanticBase):
            simple: Dict[str, str]
            semantic: List[str]

        col_descriptions = []
        for col_name, col_type in table_schema.items():
            col_type_str = col_type if isinstance(col_type, str) else col_type.get('type', 'text')
            parts = [f"- {col_name} ({col_type_str})"]
            if col_name in metadata:
                col_meta = metadata[col_name]
                if col_meta.get('description'):
                    parts.append(f"  description: {col_meta['description']}")
                for k in ('max_length', 'length', 'min_length', 'min', 'max',
                          'primary_key', 'unique', 'format', 'enum'):
                    v = col_meta.get('constraints', {}).get(k)
                    if v is not None:
                        parts.append(f"  {k}: {v}")
            col_descriptions.append("\n".join(parts))

        analysis_prompt = f"""Analyze this database schema and classify each column for synthetic data generation.

Table: {table_description or schema_name}
Columns:
{chr(10).join(col_descriptions)}

Classify each column as:
- "simple": can be generated with a Python function (IDs, dates, enums, emails, phones, codes, numbers, booleans)
- "semantic": requires LLM to be meaningful (clinical notes, diagnoses, narratives, free-text descriptions, summaries, doctor observations — any column where random text is gibberish)

For each "simple" column write a Python generator function:
  def generate_<col_name>(row, col_name):
      # may use: random, string, datetime, timedelta, uuid, math
      return <value>

Enforce all constraints in the function:
  - primary_key: use row.name + 1 (sequential integer)
  - unique: use uuid4() hex prefix
  - max_length / length: truncate or pad strings accordingly
  - min / max: clamp numeric values
  - enum: pick randomly from the list
  - format hints: emails must contain '@', dates as ISO 8601 strings

Return valid JSON exactly:
{{
  "simple": {{"col_name": "def generate_col_name(row, col_name):\\n    import random\\n    return ..."}},
  "semantic": ["col_name1", "col_name2"]
}}

Every column must appear in exactly one list."""

        agent = self.llm_client.create_agent(
            output_type=SchemaAnalysis,
            system_prompt=(
                "You are a code generation assistant that writes Python generator functions "
                "for synthetic database column values."
            ),
        )

        if ModelRetry is not None:
            @agent.output_validator
            def _validate_code(output: SchemaAnalysis) -> SchemaAnalysis:
                for col_name, code in output.simple.items():
                    try:
                        compile(code, '<string>', 'exec')
                    except SyntaxError as e:
                        raise ModelRetry(
                            f"Syntax error in generator for '{col_name}': {e}. Fix the Python code."
                        )
                return output

        model_settings = self.llm_client.get_model_settings()
        result = agent.run_sync(analysis_prompt, model_settings=model_settings)
        analysis = result.output

        # Register simple column generators
        for col_name, code in analysis.simple.items():
            ns: Dict[str, Any] = {}
            try:
                exec(code, ns)
                fn_name = f"generate_{col_name}"
                if fn_name in ns:
                    self.generator_manager.register_generator(
                        '_codegen', ns[fn_name], column_name=col_name
                    )
                else:
                    # Function may have been defined with a different name — grab the first callable
                    fns = [v for v in ns.values() if callable(v)]
                    if fns:
                        self.generator_manager.register_generator(
                            '_codegen', fns[0], column_name=col_name
                        )
            except Exception as e:
                print(f"[syda] Warning: could not register generator for '{col_name}': {e}")

        print(
            f"[syda] Code-gen analysis: {len(analysis.simple)} simple columns, "
            f"{len(analysis.semantic)} semantic columns"
        )
        return analysis.semantic

    def _generate_semantic_column(
        self,
        col_name: str,
        col_schema: Any,
        table_description: Optional[str],
        sample_size: int,
    ) -> List[str]:
        """Generate values for a semantic column using a targeted LLM call."""
        col_type = col_schema if isinstance(col_schema, str) else col_schema.get('type', 'text')
        col_desc = ""
        if isinstance(col_schema, dict):
            col_desc = col_schema.get('description', '')

        constraints = []
        if isinstance(col_schema, dict):
            c = col_schema.get('constraints', {})
            if c.get('max_length'):
                constraints.append(f"Maximum {c['max_length']} characters per value.")
            if c.get('length'):
                constraints.append(f"Exactly {c['length']} characters per value.")

        prompt = (
            f"Generate {sample_size} realistic and diverse values for column '{col_name}'.\n"
            f"Table context: {table_description or 'data table'}\n"
            f"Column type: {col_type}\n"
            + (f"Column description: {col_desc}\n" if col_desc else "")
            + (f"Constraints: {' '.join(constraints)}\n" if constraints else "")
            + f"Return a JSON array of exactly {sample_size} strings."
        )

        agent = self.llm_client.create_agent(
            output_type=List[str],
            system_prompt="You are a synthetic data generator. Generate realistic, diverse values.",
        )
        model_settings = self.llm_client.get_model_settings()
        result = agent.run_sync(prompt, model_settings=model_settings)
        return result.output

    def _run_local_generation(self, table_schema: Dict, sample_size: int) -> pd.DataFrame:
        """Fill a DataFrame using registered column generators (code-gen mode simple columns)."""
        df = pd.DataFrame(index=range(sample_size))
        for col_name in table_schema:
            df[col_name] = None

        for col_name in list(table_schema.keys()):
            if col_name in self.generator_manager.column_generators:
                try:
                    gen = self.generator_manager.column_generators[col_name]
                    df[col_name] = df.apply(lambda row, g=gen, c=col_name: g(row, c), axis=1)
                except Exception as e:
                    print(f"[syda] Warning: generator for '{col_name}' failed: {e}")
        return df

    def _enforce_uniqueness(
        self, df: pd.DataFrame, table_schema: Dict, metadata: Dict, chunk_offset: int = 0
    ) -> pd.DataFrame:
        """Ensure PKs are sequential and unique columns have no duplicates."""
        for col_name, col_meta in metadata.items():
            if col_name not in df.columns:
                continue
            constraints = col_meta.get('constraints', {})
            is_pk = bool(constraints.get('primary_key'))
            is_unique = bool(constraints.get('unique'))
            if not (is_pk or is_unique):
                continue

            col_type = table_schema.get(col_name, 'text')
            if isinstance(col_type, dict):
                col_type = col_type.get('type', 'text')

            if col_type in ('integer', 'int', 'number', 'float') and is_pk:
                df[col_name] = range(chunk_offset + 1, chunk_offset + len(df) + 1)
            elif df[col_name].duplicated().any():
                for i in range(len(df)):
                    original = str(df.iloc[i][col_name]) if df.iloc[i][col_name] is not None else ""
                    prefix = original[:20]
                    df.iloc[i, df.columns.get_loc(col_name)] = f"{prefix}_{chunk_offset + i}"
        return df

    def _apply_constraints(self, df: pd.DataFrame, table_schema: Dict, metadata: Dict) -> pd.DataFrame:
        """Post-generation safety net: truncate strings and clamp numerics to schema constraints."""
        for col_name in df.columns:
            if col_name not in metadata:
                continue
            constraints = metadata[col_name].get('constraints', {})

            max_len = constraints.get('max_length') or constraints.get('length')
            if max_len:
                df[col_name] = df[col_name].apply(
                    lambda v, ml=max_len: str(v)[:ml] if isinstance(v, str) and len(str(v)) > ml else v
                )

            col_type = table_schema.get(col_name, 'text')
            if isinstance(col_type, dict):
                col_type = col_type.get('type', 'text')
            if col_type in ('integer', 'int', 'float', 'number'):
                min_val = constraints.get('min')
                max_val = constraints.get('max')
                if min_val is not None or max_val is not None:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    if min_val is not None:
                        df[col_name] = df[col_name].clip(lower=min_val)
                    if max_val is not None:
                        df[col_name] = df[col_name].clip(upper=max_val)
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
                     sample_size: int = 10,
                     batch_size: Optional[int] = None,
                     max_retries: Optional[int] = None,
                     schema_name: Optional[str] = None,
                     stream_path: Optional[str] = None,
                     fk_cols_to_keep: Optional[set] = None) -> pd.DataFrame:
        """
        Generate synthetic data based on schema using AI.

        Args:
            table_schema: Dictionary mapping field names to types (e.g., {'name': 'text', 'age': 'number'})
            metadata: Dictionary with additional info about fields
            table_description: Optional description of the table to guide generation
            prompt: Prompt for the AI model
            sample_size: Number of samples to generate
            batch_size: Max rows per LLM call in direct mode (None = auto)
            max_retries: Max retry attempts on transient errors (None = use model_config value)
            schema_name: Schema name used for logging and code-gen context

        Returns:
            DataFrame with generated data

        Raises:
            ValueError: If the data generation fails or produces invalid results
        """
        primary_key_fields = []
        for col, col_meta in metadata.items():
            if 'constraints' in col_meta and 'primary_key' in col_meta['constraints']:
                primary_key_fields.append(col)

        # Resolve generation mode
        mode = self.model_config.generation_mode
        if mode == 'auto':
            mode = 'direct' if sample_size <= 500 else 'codegen'

        effective_retries = max_retries if max_retries is not None else self.model_config.max_retries

        # ── Code-gen mode ────────────────────────────────────────────────────
        if mode == 'codegen':
            print(f"[syda] Code-gen mode: analyzing schema and generating Python functions...")
            name = schema_name or "table"
            semantic_cols = self._analyze_and_generate_code(
                table_schema, metadata, table_description, name
            )

            df = self._run_local_generation(table_schema, sample_size)

            for col in semantic_cols:
                if col in table_schema:
                    print(f"[syda] Generating semantic column '{col}' via LLM ({sample_size} values)...")
                    values = self._generate_semantic_column(
                        col, table_schema[col], table_description, sample_size
                    )
                    if len(values) > sample_size:
                        values = values[:sample_size]
                    while len(values) < sample_size:
                        values.extend(values[:sample_size - len(values)])
                    df[col] = values[:sample_size]

            df = self._enforce_uniqueness(df, table_schema, metadata, chunk_offset=0)
            df = self._apply_constraints(df, table_schema, metadata)
            df = self._apply_type_generators(df, table_schema)
            df = self._convert_column_types(df, table_schema)
            return df

        # ── Direct mode: chunked LLM calls ───────────────────────────────────
        effective_batch = self._resolve_batch_size(sample_size, batch_size)

        def _one_chunk(sz: int, offset: int) -> pd.DataFrame:
            p = self._build_prompt(table_schema, metadata, table_description,
                                   primary_key_fields, prompt, sz)
            chunk = self._call_with_retry(
                lambda _p=p, _sz=sz: self._generate_data_with_llm(table_schema, _p, _sz),
                max_retries=effective_retries,
            )
            chunk = self._enforce_uniqueness(chunk, table_schema, metadata, chunk_offset=offset)
            chunk = self._apply_constraints(chunk, table_schema, metadata)
            chunk = self._apply_type_generators(chunk, table_schema)
            chunk = self._convert_column_types(chunk, table_schema)
            return chunk

        if sample_size <= effective_batch:
            df = _one_chunk(sample_size, 0)
            if stream_path:
                append_dataframe(df, stream_path, format=_stream_fmt(stream_path))
                df = _slim(df, fk_cols_to_keep, table_schema)
            return df

        # Multiple chunks
        n_chunks = math.ceil(sample_size / effective_batch)

        if stream_path:
            # ── Streaming path: write each chunk to disk, keep only FK cols ──
            if os.path.exists(stream_path):
                os.remove(stream_path)
            fmt = _stream_fmt(stream_path)
            fk_parts: List[pd.DataFrame] = []
            for i in range(n_chunks):
                start = i * effective_batch
                end = min(start + effective_batch, sample_size)
                print(
                    f"[syda] Generating chunk {i+1}/{n_chunks} "
                    f"(rows {start+1}–{end} of {sample_size}) → {os.path.basename(stream_path)}"
                )
                chunk = _one_chunk(end - start, start)
                append_dataframe(chunk, stream_path, format=fmt)
                if fk_cols_to_keep:
                    keep = list(fk_cols_to_keep & set(chunk.columns))
                    if keep:
                        fk_parts.append(chunk[keep])
            # Return slim DataFrame: FK columns only (enough for child-table generators)
            return pd.concat(fk_parts, ignore_index=True) if fk_parts else pd.DataFrame(columns=list(table_schema.keys()))

        # ── In-memory path: accumulate and concat ────────────────────────────
        parts: List[pd.DataFrame] = []
        for i in range(n_chunks):
            start = i * effective_batch
            end = min(start + effective_batch, sample_size)
            print(f"[syda] Generating chunk {i+1}/{n_chunks} (rows {start+1}–{end} of {sample_size})...")
            parts.append(_one_chunk(end - start, start))
        return pd.concat(parts, ignore_index=True)
