"""
Unified schema handling for the SyntheticDataGenerator class.
"""
import os
import random
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Callable, Union, Type, Any, Tuple, Set
from .schema_loader import load_schema_from_file
from .utils import create_empty_dataframe, generate_random_value
from .generator_utils import identify_schema_types, infer_values_from_documents, process_template_schemas

def _load_schema_from_file(self, file_path):
    """
    Load a schema from a JSON or YAML file.
    
    Args:
        file_path: Path to the schema file (JSON or YAML)
        
    Returns:
        Dictionary containing the schema definition
    """
    return load_schema_from_file(file_path)

def unified_generate_for_schemas(
    self,
    schemas: Dict[str, Union[Dict[str, str], str]],
    prompts: Optional[Dict[str, str]] = None,
    sample_sizes: Optional[Dict[str, int]] = None,
    output_dir: Optional[str] = None,
    default_sample_size: int = 10,
    default_prompt: str = "Generate synthetic data",
    custom_generators: Optional[Dict[str, Dict[str, Callable]]] = None,
    output_format: str = 'csv',
    document_folder: Optional[str] = None,
    document_extensions: Optional[List[str]] = None,
    infer_from_documents: bool = False
) -> Dict[str, Union[pd.DataFrame, List[str]]]:
    """
    Generate synthetic data for multiple related schemas with automatic 
    dependency resolution based on foreign key relationships.
    
    This function now supports both structured data schemas and template schemas,
    automatically determining dependencies and generation order.
    
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
    6. Processes template schemas to generate documents with placeholders replaced
    
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
    
    Template schemas are identified by the presence of a '__template__' section:
    "InvoiceTemplate": {
        "__template__": {
            "source": "templates/invoice.pdf"
        },
        ...field definitions...
    }
    
    Args:
        schemas: Dictionary mapping schema names to either:
                - Schema dictionaries (e.g., {'id': 'number', 'name': 'text'})
                - File paths to JSON or YAML schema files
        prompts: Optional dictionary mapping schema names to custom prompts
        sample_sizes: Optional dictionary mapping schema names to sample sizes
        output_dir: Optional directory to save files (one per schema)
        default_sample_size: Default number of records if not specified in sample_sizes
        default_prompt: Default prompt if not specified in prompts
        custom_generators: Optional dictionary specifying custom generators for schemas and columns
                          Format: {"SchemaName": {"column_name": generator_function}}
        output_format: Format to use when saving files ('csv' or 'json')
        document_folder: Optional path to folder containing documents to infer values from
        document_extensions: Optional list of file extensions to include
        infer_from_documents: Whether to infer values from documents
        
    Returns:
        Dictionary mapping schema names to:
        - DataFrames for structured data schemas
        - Lists of document strings for template schemas
    """
    # Initialize default parameters
    if prompts is None:
        prompts = {}
    if sample_sizes is None:
        sample_sizes = {}
    if custom_generators is None:
        custom_generators = {}
        
    # Split schemas into structured and template schemas
    structured_schemas, template_schemas = identify_schema_types(self, schemas)
    
    # Process document inference if enabled
    inferred_values = {}
    if infer_from_documents and document_folder:
        inferred_values = infer_values_from_documents(document_folder, document_extensions)
        print(f"Inferred values from documents in {document_folder}")
    
    # Process structured schemas using the original implementation
    # Generate data using the original code path for structured schemas
    structured_results = {}
    if structured_schemas:
        # Create a new dictionary with only structured schemas
        structured_results = self._generate_structured_schemas(
            schemas=structured_schemas,
            prompts=prompts,
            sample_sizes=sample_sizes,
            output_dir=output_dir,
            default_sample_size=default_sample_size,
            default_prompt=default_prompt,
            custom_generators=custom_generators,
            output_format=output_format,
            inferred_values=inferred_values
        )
    
    # Process template schemas
    template_results = {}
    if template_schemas:
        template_results = process_template_schemas(
            generator=self,
            template_schemas=template_schemas,
            structured_results=structured_results,
            sample_sizes=sample_sizes,
            default_sample_size=default_sample_size,
            output_dir=output_dir
        )
    
    # Combine results
    combined_results = {**structured_results, **template_results}
    return combined_results

def _generate_structured_schemas(
    self,
    schemas,
    prompts,
    sample_sizes,
    output_dir,
    default_sample_size,
    default_prompt,
    custom_generators,
    output_format,
    inferred_values=None
):
    """
    Process structured schemas using the original implementation.
    This is a wrapper around the existing functionality to handle structured schemas.
    """
    # Process schemas to extract schema information and metadata
    processed_schemas = {}
    schema_metadata = {}
    schema_descriptions = {}
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
                raise ValueError(f"Parent schema {parent_schema} referenced by {child_schema}.{fk_column} not found in schemas")
            
            # Skip self-references when building the dependency graph
            if parent_schema != child_schema:
                dependencies.append(parent_schema)
        
        # Store unique dependencies for this schema
        if dependencies:
            schema_dependencies[child_schema] = list(set(dependencies))
    
    # Create a directed graph of schema dependencies
    G = self._build_dependency_graph(nodes=list(processed_schemas.keys()), dependencies=schema_dependencies)
    
    # Determine generation order using topological sort
    generation_order = list(nx.topological_sort(G))
    
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
            print(f"Description: {description}")
            
            # Get the prompt and sample size for this schema
            prompt = prompts.get(schema_name, default_prompt)
            sample_size = sample_sizes.get(schema_name, default_sample_size)
            
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
            
            # Use inferred values if available
            schema_inferred_values = {}
            if inferred_values and schema_name in inferred_values:
                schema_inferred_values = inferred_values[schema_name]
                print(f"Using {len(schema_inferred_values)} inferred values for {schema_name}")
            
            # Try to use the AI generation first
            try:
                # Use the _generate_structured_data method to generate data for this schema
                print(f"Generating data for {schema_name} using _generate_structured_data")
                df = self._generate_structured_data(
                    table_schema=schema, 
                    metadata=metadata, 
                    table_description=description,
                    prompt=prompt, 
                    sample_size=sample_size
                )
                
                # Check if we have the requested sample size
                if len(df) < sample_size:
                    print(f"Warning: LLM generated only {len(df)} records instead of {sample_size} for {schema_name}")
                
                # Truncate if we got more data than needed
                if len(df) > sample_size:
                    df = df.iloc[:sample_size]
                    
            except Exception as e:
                print(f"Error generating data with LLM: {str(e)}")
                print("Falling back to empty dataframe with random values")
                
                # Create an empty dataframe with the schema's columns
                df = create_empty_dataframe(schema, sample_size)
                
                # For each column, generate random values
                for col_name, col_type in schema.items():
                    type_lower = col_type.lower() if isinstance(col_type, str) else "text"
                    df[col_name] = [generate_random_value(col_type) for _ in range(sample_size)]
            
            # Make sure IDs are unique for primary key fields (usually 'id')
            # This ensures references work correctly
            if 'id' in df.columns:
                # If numeric ID, make sure they're unique
                if pd.api.types.is_numeric_dtype(df['id']):
                    # Generate unique IDs starting from 1
                    df['id'] = list(range(1, len(df) + 1))
                # If string ID, add a unique suffix if needed
                elif pd.api.types.is_string_dtype(df['id']):
                    # Check for duplicates
                    if df['id'].duplicated().any():
                        df['id'] = [f"{val}_{i}" for i, val in enumerate(df['id'])]
            
            # Post-process foreign key columns
            if schema_name in extracted_foreign_keys:
                for field_name, (parent_schema, parent_field) in extracted_foreign_keys[schema_name].items():
                    # Handle special case for self-references (hierarchical data)
                    if parent_schema == schema_name:
                        print(f"Processing self-reference for {schema_name}.{field_name}")
                        
                        # If this field isn't in the dataframe yet, add it
                        if field_name not in df.columns:
                            df[field_name] = None
                            
                        # For self-references, we need to create a hierarchical structure
                        # Some items reference others in the same table
                        # This is common for categories, org charts, etc.
                        
                        # Special case: First item should have no parent (root)
                        if len(df) > 0:
                            values = []
                            
                            # Handle based on data size
                            if len(df) == 1:
                                # Only one record - it's the root, no parent
                                values = [None]
                            elif len(df) == 2:
                                # Two records - first is root, second refers to first
                                values = [None, df['id'].iloc[0]]
                            else:
                                # Multiple records - create a hierarchy
                                # First is root, others reference a random earlier item
                                values = [None]  # First item is the root
                                
                                # For remaining items, reference a random earlier item
                                for i in range(1, len(df)):
                                    # About 20% chance of being a root item (None parent)
                                    if random.random() < 0.2:
                                        values.append(None)  # Some items are also roots
                                    else:
                                        parent_id = df['id'].iloc[random.randint(0, i-1)]
                                        values.append(parent_id)
                            
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
            
            # Store the generated data
            results[schema_name] = df
            
            # Reset column generators for the next schema
            self.column_generators = original_column_generators.copy()
            
        # Save structured results if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each schema's data
            for schema_name, df in results.items():
                if output_format == 'csv':
                    output_file = os.path.join(output_dir, f"{schema_name.lower()}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"Saved {len(df)} records to {output_file}")
                else:
                    output_file = os.path.join(output_dir, f"{schema_name.lower()}.json")
                    df.to_json(output_file, orient='records', indent=2)
                    print(f"Saved {len(df)} records to {output_file}")
        
        # Restore original generators
        self.type_generators = original_type_generators
        self.column_generators = original_column_generators
        
    except Exception as e:
        # Restore original generators in case of error
        self.type_generators = original_type_generators
        self.column_generators = original_column_generators
        raise e
        
    return results
