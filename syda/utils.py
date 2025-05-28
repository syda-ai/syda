import inspect
import os
import yaml
import json
import pandas as pd
import random
import string
from datetime import datetime, date, timedelta
from sqlalchemy import inspect as sqla_inspect


def sqlalchemy_model_to_schema(model_class):
    """Convert a SQLAlchemy model class to a schema format compatible with the data generator.
    
    Args:
        model_class: SQLAlchemy model class
        
    Returns:
        A tuple of (table_name, table_schema, metadata) where table_schema is a dictionary
        mapping column names to column types.
    """
    # Initialize schema dict and metadata
    table_schema = {}
    metadata = {}
    
    # Get table name from model class name (overridden below for SQLAlchemy models)
    # Get table name from model class name
    table_name = model_class.__name__
    
    # Add description if available
    metadata['__description__'] = model_class.__doc__ if model_class.__doc__ else f"{table_name} data"
    
    # Check if this is a regular SQLAlchemy model or a template class
    if not hasattr(model_class, '__table__'):
        # For template classes, extract schema from class attributes
        from syda.templates import SydaTemplate
        if issubclass(model_class, SydaTemplate):
            print(f"Creating schema for template class: {model_class.__name__}")
            
            # Handle template configuration attributes with double underscores
            # Add them to metadata rather than schema fields
            if hasattr(model_class, '__template_source__'):
                metadata['__template_source__'] = getattr(model_class, '__template_source__')
                # Also add as field to make template processing work
                table_schema['template_source'] = 'text'
                
            if hasattr(model_class, '__input_file_type__'):
                metadata['__input_file_type__'] = getattr(model_class, '__input_file_type__')
                # Also add as field to make template processing work
                table_schema['input_file_type'] = 'text'
                
            if hasattr(model_class, '__output_file_type__'):
                metadata['__output_file_type__'] = getattr(model_class, '__output_file_type__')
                # Also add as field to make template processing work
                table_schema['output_file_type'] = 'text'
            
            # Add special metadata attributes
            if hasattr(model_class, '__template__'):
                metadata['__template__'] = getattr(model_class, '__template__')
            if hasattr(model_class, '__depends_on__'):
                metadata['__depends_on__'] = getattr(model_class, '__depends_on__')
            
            # Get regular fields from class attributes (not starting with __)
            for attr_name, attr_value in model_class.__dict__.items():
                if not attr_name.startswith('__') and not attr_name.startswith('_') and not callable(attr_value):
                    # Add each field to the schema
                    table_schema[attr_name] = 'text'  # Default type
        
        return table_name, table_schema, metadata
    
    # For regular SQLAlchemy models with tables
    # Extract table columns
    mapper = sqla_inspect(model_class)
    for column in mapper.columns:
        # Get column type
        column_type = column.type.__class__.__name__.lower()
        
        # Map SQLAlchemy types to schema types
        if column_type in ('integer', 'biginteger', 'smallinteger'):
            table_schema[column.name] = 'integer'
        elif column_type in ('float', 'numeric', 'decimal'):
            table_schema[column.name] = 'float'
        elif column_type == 'boolean':
            table_schema[column.name] = 'boolean'
        elif column_type == 'date':
            table_schema[column.name] = 'date'
        elif column_type == 'datetime':
            table_schema[column.name] = 'datetime'
        else:
            table_schema[column.name] = 'text'
    
    # Use actual table name from SQLAlchemy model
    if hasattr(model_class, '__tablename__'):
        table_name = getattr(model_class, '__tablename__')
        print(f"Using SQLAlchemy table name: {table_name}")
    
    # Extract foreign keys
    foreign_keys = {}
    for column in mapper.columns:
        if column.foreign_keys:
            # Get foreign key reference
            for fk in column.foreign_keys:
                target_table = fk.column.table.name
                target_column = fk.column.name
                foreign_keys[column.name] = {
                    'references': f"{target_table}.{target_column}"
                }
    
    # Add foreign keys to metadata if any exist
    if foreign_keys:
        metadata['__foreign_keys__'] = foreign_keys
    
    return table_name, table_schema, metadata


def create_empty_dataframe(schema):
    """Create an empty pandas DataFrame with columns matching the schema types."""
    columns = {}
    for field, field_type in schema.items():
        # Skip metadata fields
        if field.startswith('__') and field.endswith('__'):
            continue
        # Map schema types to pandas dtypes
        if field_type == 'integer':
            columns[field] = pd.Series(dtype='int64')
        elif field_type == 'float':
            columns[field] = pd.Series(dtype='float64')
        elif field_type == 'boolean':
            columns[field] = pd.Series(dtype='bool')
        elif field_type in ('date', 'datetime'):
            columns[field] = pd.Series(dtype='datetime64[ns]')
        else:
            columns[field] = pd.Series(dtype='object')
    
    return pd.DataFrame(columns)


def generate_random_value(field_type):
    """Generate a random value based on field type for placeholder data."""
    if field_type == 'integer':
        return random.randint(1, 1000)
    elif field_type == 'float':
        return round(random.uniform(1.0, 1000.0), 2)
    elif field_type == 'boolean':
        return random.choice([True, False])
    elif field_type == 'date':
        # Random date in last 5 years
        days = random.randint(0, 365 * 5)
        return (date.today() - timedelta(days=days)).isoformat()
    elif field_type == 'datetime':
        # Random datetime in last 5 years
        days = random.randint(0, 365 * 5)
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        seconds = random.randint(0, 59)
        dt = datetime.now() - timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return dt.isoformat()
    else:  # text or any other type
        # Generate random string
        length = random.randint(5, 15)
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def get_schema_prompt(schema, table_name, description=None):
    """Generate a prompt for the LLM based on schema information."""
    prompt = f"Generate data for {table_name}"
    if description:
        prompt += f": {description}"
    return prompt


def parse_dataframe_output(text, schema):
    """Parse LLM output text into a pandas DataFrame based on schema."""
    try:
        # Try to parse as JSON
        data = json.loads(text)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            # If not a valid JSON structure, raise error
            raise ValueError("Output is not a valid JSON structure")
        
        # Type conversion based on schema
        for col, dtype in schema.items():
            if col in df.columns:
                if dtype == 'integer':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                elif dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                elif dtype == 'boolean':
                    df[col] = df[col].astype(bool)
                elif dtype in ('date', 'datetime'):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error parsing output: {e}")
        # Return empty DataFrame matching schema
        return create_empty_dataframe(schema)


def read_schema_file(schema_file):
    """Read a schema file (JSON or YAML) and return the schema dictionary."""
    try:
        with open(schema_file, 'r') as file:
            if schema_file.endswith('.json'):
                return json.load(file)
            elif schema_file.endswith(('.yaml', '.yml')):
                return yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported schema file format: {schema_file}")
    except Exception as e:
        print(f"Error reading schema file {schema_file}: {e}")
        return {}


def save_dataframe(df, output_file):
    """Save a dataframe to a file (CSV, Excel, JSON, or Parquet)."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save based on file extension
    file_ext = os.path.splitext(output_file)[1].lower()
    
    if file_ext == '.csv':
        df.to_csv(output_file, index=False)
        print(f"✓ Successfully wrote {len(df)} rows to {output_file}")
    elif file_ext in ('.xls', '.xlsx'):
        df.to_excel(output_file, index=False)
        print(f"✓ Successfully wrote {len(df)} rows to {output_file}")
    elif file_ext == '.json':
        df.to_json(output_file, orient='records', lines=True)
        print(f"✓ Successfully wrote {len(df)} rows to {output_file}")
    elif file_ext == '.parquet':
        df.to_parquet(output_file, index=False)
        print(f"✓ Successfully wrote {len(df)} rows to {output_file}")
    else:
        print(f"⚠️ Unsupported file format: {file_ext}. Defaulting to CSV.")
        csv_file = os.path.splitext(output_file)[0] + '.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ Successfully wrote {len(df)} rows to {csv_file}")
