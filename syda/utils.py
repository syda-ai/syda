"""
Utility functions for the syda package.
Contains helper functions for data conversion, type mapping, and SQLAlchemy integration.
"""

from typing import Dict, Optional, List, Union, Callable, Tuple, Type, Any
import random
import pandas as pd
import json

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
            elif py_type == int:
                schema[col.name] = 'number'
            elif py_type == float:
                schema[col.name] = 'number'
            elif py_type == bool:
                schema[col.name] = 'boolean'
            elif py_type.__name__ == 'date':
                schema[col.name] = 'date'
            elif py_type.__name__ == 'datetime':
                schema[col.name] = 'datetime'
            else:
                schema[col.name] = 'text'  # Default to text
        else:
            schema[col.name] = 'text'  # Default to text
        
        # Collect metadata
        col_metadata = {}
        
        # Add comment if available
        if col.comment:
            col_metadata['comment'] = col.comment
        
        # Add constraints
        constraints = []
        if col.primary_key:
            constraints.append('primary_key')
        if col.unique:
            constraints.append('unique')
        if not col.nullable:
            constraints.append('not_null')
        if col.foreign_keys:
            # Get foreign key target table and column
            for fk in col.foreign_keys:
                target = fk.target_fullname
                constraints.append(f'foreign_key_to({target})')
        
        if constraints:
            col_metadata['constraints'] = constraints
        
        # Store metadata
        if col_metadata:
            metadata[col.name] = col_metadata
    
    return schema, metadata, model_docstring

def extract_sqlalchemy_relationships(model_class) -> Dict[str, Dict]:
    """
    Extract relationship information from a SQLAlchemy model.
    
    Returns:
        dict: Dictionary of relationship name to target model details
    """
    relationships = {}
    
    if not hasattr(model_class, '__mapper__') or not hasattr(model_class.__mapper__, 'relationships'):
        return relationships
    
    for relationship_name, relationship in model_class.__mapper__.relationships.items():
        target_model = relationship.argument
        if callable(target_model):
            target_model = target_model()
        
        # Get target model name
        if hasattr(target_model, '__name__'):
            target_model_name = target_model.__name__
        else:
            target_model_name = str(target_model)
        
        relationships[relationship_name] = {
            'target_model': target_model_name,
            'direction': 'many_to_one' if relationship.direction.name == 'MANYTOONE' else 
                        'one_to_many' if relationship.direction.name == 'ONETOMANY' else 
                        'many_to_many' if relationship.direction.name == 'MANYTOMANY' else 'one_to_one'
        }
    
    return relationships

def create_empty_dataframe(schema: Dict) -> pd.DataFrame:
    """
    Create an empty DataFrame with columns based on the schema.
    
    Args:
        schema: Dictionary mapping column names to column types
        
    Returns:
        pd.DataFrame: Empty DataFrame with appropriate columns
    """
    return pd.DataFrame({k: [] for k in schema.keys()})

def generate_random_value(col_type: str) -> Any:
    """
    Generate a random value for a given column type.
    
    Args:
        col_type: Type of column (text, number, boolean, date, datetime)
        
    Returns:
        Any: Random value appropriate for the column type
    """
    if col_type == 'text':
        return f"value_{random.randint(1, 1000)}"
    elif col_type == 'number':
        return random.randint(1, 1000)
    elif col_type == 'boolean':
        return random.choice([True, False])
    elif col_type == 'date':
        return f"{random.randint(2000, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
    elif col_type == 'datetime':
        return f"{random.randint(2000, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
    else:
        return f"value_{random.randint(1, 1000)}"  # Default to text-like

def get_schema_prompt(schema: Dict, metadata: Optional[Dict] = None, model_docstring: Optional[str] = None) -> str:
    """
    Generate a prompt describing the schema for the LLM.
    
    Args:
        schema: Dictionary mapping column names to column types
        metadata: Optional dictionary of column metadata
        model_docstring: Optional model docstring
        
    Returns:
        str: Formatted prompt section describing the schema
    """
    prompt_parts = []
    
    # Add model description if available
    if model_docstring:
        prompt_parts.append(f"Model Description: {model_docstring}\n")
    
    prompt_parts.append("The schema has the following columns:")
    
    # Process each column
    for col_name, col_type in schema.items():
        col_desc = f"- {col_name} ({col_type})"
        
        # Add metadata if available
        if metadata and col_name in metadata:
            col_meta = metadata[col_name]
            
            # Add comment
            if 'comment' in col_meta:
                col_desc += f": {col_meta['comment']}"
            
            # Add constraints
            if 'constraints' in col_meta:
                constraints = ", ".join(col_meta['constraints'])
                col_desc += f" [Constraints: {constraints}]"
                
        prompt_parts.append(col_desc)
    
    return "\n".join(prompt_parts)

def parse_dataframe_output(df_output: str) -> pd.DataFrame:
    """
    Parse DataFrame output from LLM response.
    
    Args:
        df_output: String representation of dataframe data (JSON, CSV, etc.)
        
    Returns:
        pd.DataFrame: Parsed DataFrame
    """
    # Try parsing as JSON
    try:
        # Clean up common issues in JSON responses
        clean_output = df_output.strip()
        
        # Handle triple backticks in code blocks
        if clean_output.startswith("```") and clean_output.endswith("```"):
            # Extract just the JSON part
            clean_output = clean_output[clean_output.find("\n")+1:clean_output.rfind("```")].strip()
        
        # Handle if json keyword is included
        if clean_output.startswith("json"):
            clean_output = clean_output[4:].strip()
            
        # Parse the JSON
        data = json.loads(clean_output)
        
        # Handle various JSON formats
        if isinstance(data, list):
            # List of records
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], list):
                # Some LLMs wrap the data in a 'data' key
                return pd.DataFrame(data['data'])
            elif all(isinstance(v, list) for v in data.values()):
                # Columns as lists
                return pd.DataFrame(data)
            else:
                # Single record
                return pd.DataFrame([data])
        else:
            # Fallback empty DataFrame
            return pd.DataFrame()
    except Exception as e:
        # Fallback to empty DataFrame on parsing errors
        print(f"Error parsing LLM output as DataFrame: {str(e)}")
        return pd.DataFrame()
