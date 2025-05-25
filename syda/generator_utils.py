"""
Utility functions for the unified data generation system.
"""
import os
from typing import Dict, List, Optional, Union, Any
import pandas as pd

def identify_schema_types(generator, schemas):
    """
    Identify which schemas are structured and which are templates.
    
    Args:
        generator: SyntheticDataGenerator instance
        schemas: Dictionary mapping schema names to schema sources
        
    Returns:
        Tuple of (structured_schemas, template_schemas)
    """
    structured_schemas = {}
    template_schemas = {}
    
    for schema_name, schema_source in schemas.items():
        # Load schema if it's a file path
        schema_dict = schema_source
        if isinstance(schema_source, str) and (schema_source.endswith('.json') or 
                                           schema_source.endswith('.yml') or 
                                           schema_source.endswith('.yaml')):
            schema_dict = generator._load_schema_from_file(schema_source)
        
        # Check if this is a template schema
        if isinstance(schema_dict, dict) and generator._is_template_schema(schema_dict):
            template_schemas[schema_name] = schema_dict
        else:
            structured_schemas[schema_name] = schema_source
            
    return structured_schemas, template_schemas

def infer_values_from_documents(document_folder, extensions=None):
    """
    Extract values from documents in the specified folder.
    
    Args:
        document_folder: Path to folder containing documents
        extensions: Optional list of file extensions to include
        
    Returns:
        Dictionary mapping schema names to inferred values
    """
    if not os.path.exists(document_folder):
        print(f"Document folder {document_folder} does not exist")
        return {}
        
    # Default extensions if none provided
    if extensions is None:
        extensions = ['.pdf', '.docx', '.doc', '.txt']
        
    # For now, just return an empty dictionary
    # In a real implementation, we would process documents and extract values
    return {}

def process_template_schemas(
    generator,
    template_schemas: Dict[str, Dict],
    structured_results: Dict[str, pd.DataFrame],
    sample_sizes: Dict[str, int],
    default_sample_size: int = 10,
    output_dir: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Process template schemas to generate documents.
    
    Args:
        generator: SyntheticDataGenerator instance
        template_schemas: Dictionary mapping schema names to template schema dictionaries
        structured_results: Dictionary of structured data results
        sample_sizes: Dictionary mapping schema names to sample sizes
        default_sample_size: Default sample size if not specified
        output_dir: Optional directory to save output files
        
    Returns:
        Dictionary mapping schema names to lists of document strings
    """
    template_results = {}
    
    for schema_name, schema_dict in template_schemas.items():
        # Get sample size for this template
        sample_size = sample_sizes.get(schema_name, default_sample_size)
        
        # Process the template schema to generate documents
        try:
            documents = generator._process_template_schema(
                schema_name=schema_name,
                schema_dict=schema_dict,
                structured_data=structured_results,
                sample_size=sample_size
            )
            
            # Store the generated documents
            template_results[schema_name] = documents
            
            # Save template documents if output_dir is provided
            if output_dir:
                template_dir = os.path.join(output_dir, schema_name)
                os.makedirs(template_dir, exist_ok=True)
                
                # Save each document
                for i, doc in enumerate(documents):
                    output_file = os.path.join(template_dir, f"document_{i+1}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(doc)
                    print(f"Saved template document to {output_file}")
        
        except Exception as e:
            print(f"Error processing template {schema_name}: {str(e)}")
    
    return template_results
    
def identify_schema_types(
    generator,
    schemas: Dict[str, Union[Dict[str, Any], str]]
) -> tuple:
    """
    Identify which schemas are structured and which are templates.
    
    Args:
        generator: SyntheticDataGenerator instance
        schemas: Dictionary mapping schema names to schema sources
        
    Returns:
        Tuple of (structured_schemas, template_schemas)
    """
    structured_schemas = {}
    template_schemas = {}
    
    for schema_name, schema_source in schemas.items():
        # Load schema if it's a file path
        schema_dict = schema_source
        if isinstance(schema_source, str) and (schema_source.endswith('.json') or 
                                           schema_source.endswith('.yml') or 
                                           schema_source.endswith('.yaml')):
            schema_dict = generator._load_schema_from_file(schema_source)
        
        # Check if this is a template schema
        if isinstance(schema_dict, dict) and generator._is_template_schema(schema_dict):
            template_schemas[schema_name] = schema_dict
        else:
            structured_schemas[schema_name] = schema_source
            
    return structured_schemas, template_schemas
    
def infer_values_from_documents(
    document_folder: str,
    extensions: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[Any]]]:
    """
    Extract values from documents in the specified folder.
    
    Args:
        document_folder: Path to folder containing documents
        extensions: Optional list of file extensions to include
        
    Returns:
        Dictionary mapping schema names to inferred values
    """
    if not os.path.exists(document_folder):
        print(f"Document folder {document_folder} does not exist")
        return {}
        
    # Default extensions if none provided
    if extensions is None:
        extensions = ['.pdf', '.docx', '.doc', '.txt']
        
    # For now, just return an empty dictionary
    # In a real implementation, we would process documents and extract values
    return {}
