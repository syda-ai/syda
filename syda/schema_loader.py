"""
Schema loading functionality for the SyntheticDataGenerator.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any

def load_schema_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load a schema from a JSON or YAML file.
    
    Args:
        file_path: Path to the schema file (JSON or YAML)
        
    Returns:
        Dictionary containing the schema definition
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Schema file not found: {file_path}")
        
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.json':
        # Load JSON schema
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_extension in ['.yml', '.yaml']:
        # Load YAML schema
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML schema files. Install with 'pip install pyyaml'")
    else:
        raise ValueError(f"Unsupported schema file format: {file_extension}")
