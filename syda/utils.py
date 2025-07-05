import inspect
import os
import yaml
import json
import pandas as pd
import random
import string
from datetime import datetime, date, timedelta
from sqlalchemy import inspect as sqla_inspect, Column
from typing import Dict, Optional, Any, Union

def create_empty_dataframe(schema: Dict[str, str]) -> pd.DataFrame:
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


def generate_random_value(field_type: str) -> Any:
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


def get_schema_prompt(
    schema: Dict[str, str],
    table_name: str,
    description: Optional[str] = None
) -> str:
    """Generate a prompt for the LLM based on schema information."""
    prompt = f"Generate data for {table_name}"
    if description:
        prompt += f": {description}"
    return prompt


def parse_dataframe_output(
    text: str,
    schema: Dict[str, str]
) -> pd.DataFrame:
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


def save_dataframe(
    df: pd.DataFrame,
    output_file: str
) -> str:
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
