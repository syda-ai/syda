"""
Schema loading and processing for synthetic data generation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union, Type, Any
from .schemas import validate_schema

# Check if yaml is available
try:
    import yaml
    YAML_INSTALLED = True
except ImportError:
    YAML_INSTALLED = False

# Check if SQLAlchemy is available
try:
    from sqlalchemy.inspection import inspect as sqla_inspect
    SQLALCH_INSTALLED = True
except ImportError:
    SQLALCH_INSTALLED = False

class SchemaLoader:
    """
    Load and process schemas in various formats:
    - Dictionary schemas
    - JSON/YAML schema files
    - SQLAlchemy models
    - Template classes
    
    Handles loading, normalization, metadata extraction, and foreign key parsing.
    """
    
    def __init__(self):
        """Initialize the schema loader."""
        pass
        
    def load_schema(
        self, 
        schema_source: Union[Dict, str, Type], 
        schema_name: Optional[str] = None) -> Tuple[Dict, Dict, Optional[str], Dict, List]:
        """
        Load and process a schema from various source formats.
        
        Args:
            schema_source: Either a dictionary mapping field names to types,
                      a file path to a JSON/YAML schema file, a SQLAlchemy model class, or a template class
            schema_name: Optional name of the schema, used for error reporting
            
        Returns:
            Tuple of (table_schema, metadata, table_description, foreign_keys)
            
            Example return values:
                - table_schema: {'id': 'number', 'name': 'text', 'email': 'email', 'created_at': 'date'}
                - metadata: {
                    'id': {'description': 'Primary key', 'constraints': {'primary_key': True}},
                    'name': {'description': 'User full name', 'constraints': {'max_length': 100}},
                    'email': {'description': 'User email address', 'constraints': {'unique': True}},
                    'created_at': {'description': 'Account creation date'}
                }
                - table_description: "User account information for the system"
                - foreign_keys: {'user_id': ('User', 'id')}
                - depends_on_schemas: ['User', 'Department']
        """
        # Case 1: Dictionary schema
        if isinstance(schema_source, dict):
            try:
                validate_schema(schema_source)
            except ValueError as e:
                # For dictionary schemas, include schema name if available or field count for better context
                schema_name = schema_source.get('__name__', 'Unknown schema')
                field_count = len([k for k in schema_source.keys() if not k.startswith('__')])
                raise ValueError(f"Schema validation failed for dictionary schema '{schema_name}' with {field_count} fields: {str(e)}")
            return self._load_dict_schema(schema_source)
        
        # Case 2: SQLAlchemy model - check for __table__ attribute which all SQLAlchemy models have
        elif SQLALCH_INSTALLED and isinstance(schema_source, type) and hasattr(schema_source, '__table__'):
            return self._load_sqlalchemy_model(schema_source)
                
        # Case 3: Path to JSON/YAML schema file
        elif isinstance(schema_source, str):
            if not os.path.exists(schema_source):
                raise ValueError(f"Schema file not found: {schema_source}")
            schema_dict = self._load_schema_file(schema_source)
            print("schema_dict", schema_dict)
           # exit(0)
            try:
                validate_schema(schema_dict)
            except ValueError as e:
                # Enhance error message with file path information
                raise ValueError(f"Schema validation failed for '{schema_source}': {str(e)}")
            return self._load_dict_schema(schema_dict)
        
        # Case 4: Unsupported type
        else:
            schema_type = type(schema_source).__name__
            raise ValueError(f"Unsupported schema type: {schema_type} for schema {schema_name or 'unknown'}")
    
    def _load_dict_schema(self, schema_dict: Dict) -> Tuple[Dict, Dict, Optional[str], Dict, List]:
        """
        Process a dictionary schema.
        
        Args:
            schema_dict: Dictionary schema definition
            
        Returns:
            Tuple of (table_schema, metadata, table_description, foreign_keys, depends_on_schemas)
        """
        table_schema = {}
        metadata = {}
        table_description = None
        foreign_keys = {}
        depends_on_schemas = []
        # Extract table description if present
        if "__description__" in schema_dict:
            table_description = schema_dict["__description__"]
        elif "__table_description__" in schema_dict:
            table_description = schema_dict["__table_description__"]
        if "__depends_on__" in schema_dict:
            depends_on_schemas = schema_dict["__depends_on__"]
        
        # Extract foreign keys if present
        if "__foreign_keys__" in schema_dict:
            for fk_col, fk_ref in schema_dict["__foreign_keys__"].items():
                if isinstance(fk_ref, list) and len(fk_ref) == 2:
                    # Convert lists to tuples for consistency
                    foreign_keys[fk_col] = (fk_ref[0], fk_ref[1])
                else:
                    foreign_keys[fk_col] = fk_ref
        template_fields = {}
        # Extract fields and metadata
        for field_name, field_info in schema_dict.items():
            # Skip special fields with double underscores
            if field_name.startswith("__"):
                if field_name in ["__template__", "__template_source__", "__input_file_type__", "__output_file_type__"]:
                    template_fields[field_name] = field_info
                continue
            
            # Handle simple field definition (field: "type")
            if isinstance(field_info, str):
                table_schema[field_name] = field_info
                metadata[field_name] = {}
            # Handle complex field definition with metadata
            elif isinstance(field_info, dict):
                field_type = field_info.get("type", "text")
                table_schema[field_name] = field_type
                
                # Extract field metadata
                field_metadata = {}
                
                # Add description if available
                if "description" in field_info:
                    field_metadata["description"] = field_info["description"]
                
                # Extract foreign key relationship
                if field_type == "foreign_key" and "references" in field_info:
                    references = field_info["references"]
                    if isinstance(references, dict) and "schema" in references and "field" in references:
                        foreign_keys[field_name] = (references["schema"], references["field"])
                    elif isinstance(references, str) and "." in references:
                        # Handle string format like "Table.column"
                        parts = references.split(".")
                        if len(parts) == 2:
                            foreign_keys[field_name] = (parts[0], parts[1])
                
                # Process constraints
                constraints = {}
                
                # Extract direct constraint attributes
                for constraint_key in ["nullable", "primary_key", "unique", "length", "max_length", "min_length"]:
                    if constraint_key in field_info:
                        if "constraints" not in field_metadata:
                            field_metadata["constraints"] = {}
                        field_metadata["constraints"][constraint_key] = field_info[constraint_key]
                
                # Process explicit constraints section
                if "constraints" in field_info:
                    if "constraints" not in field_metadata:
                        field_metadata["constraints"] = {}
                    for k, v in field_info["constraints"].items():
                        field_metadata["constraints"][k] = v
                
                # Add field metadata if any was collected
                if field_metadata:
                    metadata[field_name] = field_metadata
        
        return table_schema, metadata, table_description, foreign_keys, template_fields, depends_on_schemas
    
    def _load_schema_file(self, file_path: str) -> Tuple[Dict, Dict, Optional[str], Dict]:
        """
        Process a schema file (JSON or YAML).
        
        Args:
            file_path: Path to schema file
            
        Returns:
            Tuple of (table_schema, metadata, table_description, foreign_keys)
        """
        # Load file based on extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # JSON file
        if file_ext in ['.json', '.schema']:
            try:
                with open(file_path, 'r') as f:
                    schema_dict = json.load(f)
                    return schema_dict
            except Exception as e:
                raise ValueError(f"Error loading JSON schema file {file_path}: {str(e)}")
                
        # YAML file
        elif file_ext in ['.yml', '.yaml']:
            if not YAML_INSTALLED:
                raise ImportError("PyYAML is required for YAML schema support. Install it with 'pip install pyyaml'.")
            
            try:
                with open(file_path, 'r') as f:
                    schema_dict = yaml.safe_load(f)
                    return schema_dict
            except Exception as e:
                raise ValueError(f"Error loading YAML schema file {file_path}: {str(e)}")
        
        # Unsupported file type
        else:
            raise ValueError(f"Unsupported schema file type: {file_ext} for file {file_path}")
         
    def _load_sqlalchemy_model(self, model_class: Type) -> Dict:
        """
        Process a SQLAlchemy model class. Can also process template models that extend the Base class.
        
        Args:
            model_class: SQLAlchemy model class (or template model class)
            
        Returns:
            Dictionary containing the schema definition
        """
        if not SQLALCH_INSTALLED:
            raise ImportError("SQLAlchemy is required but not installed. Install it with 'pip install sqlalchemy'.")
        
        schema_dict = {}
        foreign_keys = {}
        
        # Get table name from model
        if hasattr(model_class, '__table__'):
            table_name = model_class.__table__.name
        else:
            table_name = model_class.__name__
        schema_dict['__table_name__'] = table_name
        
        # Get docstring as description
        table_description = model_class.__doc__ or f"{table_name} data"
        schema_dict['__description__'] = table_description
        
        # Check if this is a template model
        is_template = hasattr(model_class, '__template__') and model_class.__template__
        
        # Add template-specific flag and attributes if it's a template model
        if is_template:
            schema_dict['__template__'] = True
            
            # Get template-specific attributes
            if hasattr(model_class, '__template_source__'):
                schema_dict['__template_source__'] = model_class.__template_source__
                
            if hasattr(model_class, '__input_file_type__'):
                schema_dict['__input_file_type__'] = model_class.__input_file_type__
                
            if hasattr(model_class, '__output_file_type__'):
                schema_dict['__output_file_type__'] = model_class.__output_file_type__
        
        # Get dependencies if defined
        if hasattr(model_class, '__depends_on__'):
            schema_dict['__depends_on__'] = model_class.__depends_on__
        
        # Process columns using SQLAlchemy inspection if possible
        columns = []
        try:
            # Use SQLAlchemy inspection to get columns
            mapper = sqla_inspect(model_class)
            columns = mapper.columns
        except Exception as e:
            # If inspection fails (e.g., for template models without proper mapping),
            # try to find columns as attributes
            for attr_name in dir(model_class):
                if not attr_name.startswith('_') and not callable(getattr(model_class, attr_name)):
                    attr = getattr(model_class, attr_name)
                    if hasattr(attr, 'type') and hasattr(attr.type, 'python_type'):
                        columns.append(attr)
        
        # Process each column
        for column in columns:
            # Skip if this isn't actually a column
            if not hasattr(column, 'type') or not hasattr(column.type, 'python_type'):
                continue
                
            column_name = getattr(column, 'name', None) or getattr(column, 'key', None)
            if not column_name:
                continue
                
            # Get column type
            column_type = column.type.__class__.__name__.lower()
            
            # Initialize field dictionary
            field_dict = {}
            
            # Map SQLAlchemy types to schema types
            if column_type in ('integer', 'biginteger', 'smallinteger'):
                field_dict['type'] = 'integer'
            elif column_type in ('float', 'numeric', 'decimal'):
                field_dict['type'] = 'float'
            elif column_type == 'boolean':
                field_dict['type'] = 'boolean'
            elif column_type == 'date':
                field_dict['type'] = 'date'
            elif column_type == 'datetime':
                field_dict['type'] = 'datetime'
            elif column_type in ('text', 'string', 'unicode', 'varchar'):
                field_dict['type'] = 'text'
            elif column_type == 'json':
                field_dict['type'] = 'json'
            else:
                # Default to text for unknown types
                field_dict['type'] = 'text'
            
            # Add description if available (from comment)
            if hasattr(column, 'comment') and column.comment:
                field_dict['description'] = column.comment
            
            # Handle foreign keys
            if hasattr(column, 'foreign_keys') and column.foreign_keys:
                try:
                    for fk in column.foreign_keys:
                        if hasattr(fk, 'column') and fk.column is not None:
                            # Extract table and column name from foreign key target
                            target_table = fk.column.table.name
                            target_column = fk.column.name
                            foreign_keys[column_name] = (target_table, target_column)
                            # Mark field as foreign_key type
                            field_dict['type'] = 'foreign_key'
                            break
                except Exception as e:
                    # Try to infer foreign key from name (e.g., user_id -> users.id)
                    if column_name.endswith('_id'):
                        target_entity = column_name[:-3]  # Remove '_id' suffix
                        foreign_keys[column_name] = (target_entity + 's', 'id')  # Assume plural table name
                        field_dict['type'] = 'foreign_key'
            
            # Add constraints
            constraints = {}
            
            # Check for primary key
            if hasattr(column, 'primary_key') and column.primary_key:
                constraints['primary_key'] = True
            
            # Check for nullable
            if hasattr(column, 'nullable'):
                if not column.nullable:
                    constraints['not_null'] = True
            
            # Check for unique constraint
            if hasattr(column, 'unique') and column.unique:
                constraints['unique'] = True
            
            # Add length constraint for string types
            if hasattr(column.type, 'length') and column.type.length is not None:
                constraints['length'] = column.type.length
            
            # Add numeric constraints if available
            if hasattr(column.type, 'precision') and column.type.precision is not None:
                constraints['precision'] = column.type.precision
            if hasattr(column.type, 'scale') and column.type.scale is not None:
                constraints['scale'] = column.type.scale
                
            # Add min/max if defined in column info
            if hasattr(column, 'info'):
                if 'min' in column.info:
                    constraints['min'] = column.info['min']
                if 'max' in column.info:
                    constraints['max'] = column.info['max']
                    
            if constraints:
                field_dict['constraints'] = constraints
                
            # Add field to schema dictionary
            schema_dict[column_name] = field_dict
            
        # Handle template-specific foreign keys extraction if it's a template model
        if is_template and hasattr(model_class, 'get_foreign_keys') and callable(getattr(model_class, 'get_foreign_keys')):
            try:
                template_foreign_keys = model_class.get_foreign_keys()
                for fk_field, fk_info in template_foreign_keys.items():
                    # Update existing field to be a foreign key
                    if fk_field in schema_dict:
                        schema_dict[fk_field]['type'] = 'foreign_key'
                    
                    # Add foreign key relationship
                    foreign_keys[fk_field] = (fk_info['target_table'], fk_info['target_column'])
                    print(f"Using template-defined foreign key: {fk_field} -> {fk_info['target_table']}.{fk_info['target_column']}")
            except Exception as e:
                print(f"Warning: Error extracting foreign keys from template class: {e}")
            
        # Add foreign keys to schema if any
        if foreign_keys:
            schema_dict['__foreign_keys__'] = foreign_keys
            
        return schema_dict