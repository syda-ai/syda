"""
Schema loading and processing for synthetic data generation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union, Type, Any

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
        
    def load_schema(self, schema_source: Union[Dict, str, Type], schema_name: Optional[str] = None) -> Tuple[Dict, Dict, Optional[str], Dict]:
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
        """
        # Case 1: Dictionary schema
        if isinstance(schema_source, dict):
            return self._load_dict_schema(schema_source)
        
        # Case 2: SQLAlchemy model - check for __table__ attribute which all SQLAlchemy models have
        elif SQLALCH_INSTALLED and isinstance(schema_source, type) and hasattr(schema_source, '__table__'):
            return self._load_sqlalchemy_model(schema_source)
                
        # Case 3: Path to JSON/YAML schema file
        elif isinstance(schema_source, str):
            if os.path.exists(schema_source):
                return self._load_schema_file(schema_source)
            else:
                raise ValueError(f"Schema file not found: {schema_source}")
        
        # Case 4: Template class - check for __template__ attribute or SydaTemplate inheritance
        elif isinstance(schema_source, type) and (
            hasattr(schema_source, '__template__') or 
            any('SydaTemplate' == base.__name__ for base in schema_source.__mro__ if hasattr(base, '__name__'))
        ):
            return self._load_template_class(schema_source)
        
        # Case 5: Unsupported type
        else:
            schema_type = type(schema_source).__name__
            raise ValueError(f"Unsupported schema type: {schema_type} for schema {schema_name or 'unknown'}")
    
    def _load_dict_schema(self, schema_dict: Dict) -> Tuple[Dict, Dict, Optional[str], Dict]:
        """
        Process a dictionary schema.
        
        Args:
            schema_dict: Dictionary schema definition
            
        Returns:
            Tuple of (table_schema, metadata, table_description, foreign_keys)
        """
        table_schema = {}
        metadata = {}
        table_description = None
        foreign_keys = {}
        
        # Extract table description if present
        if "__description__" in schema_dict:
            table_description = schema_dict["__description__"]
        elif "__table_description__" in schema_dict:
            table_description = schema_dict["__table_description__"]
        
        # Extract foreign keys if present
        if "__foreign_keys__" in schema_dict:
            for fk_col, fk_ref in schema_dict["__foreign_keys__"].items():
                if isinstance(fk_ref, list) and len(fk_ref) == 2:
                    # Convert lists to tuples for consistency
                    foreign_keys[fk_col] = (fk_ref[0], fk_ref[1])
                else:
                    foreign_keys[fk_col] = fk_ref
        
        # Extract fields and metadata
        for field_name, field_info in schema_dict.items():
            # Skip special fields with double underscores
            if field_name.startswith("__"):
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
        
        return table_schema, metadata, table_description, foreign_keys
    
    def _load_sqlalchemy_model(self, model_class: Type) -> Tuple[Dict, Dict, Optional[str], Dict]:
        """
        Process a SQLAlchemy model class.
        
        Args:
            model_class: SQLAlchemy model class
            
        Returns:
            Tuple of (table_schema, metadata, table_description, foreign_keys)
        """
        if not SQLALCH_INSTALLED:
            raise ImportError("SQLAlchemy is required but not installed. Install it with 'pip install sqlalchemy'.")
        
        table_schema = {}
        metadata = {}
        foreign_keys = {}
        
        # Get table name from model
        table_name = model_class.__table__.name
        metadata['__table_name__'] = table_name
        
        # Get docstring as description
        table_description = model_class.__doc__ or f"{table_name} data"
        
        # Use SQLAlchemy inspection to get columns
        mapper = sqla_inspect(model_class)
        
        # Process each column
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
            elif column_type in ('text', 'string', 'unicode', 'varchar'):
                table_schema[column.name] = 'text'
            elif column_type == 'json':
                table_schema[column.name] = 'json'
            else:
                # Default to text for unknown types
                table_schema[column.name] = 'text'
            
            # Add column metadata
            field_metadata = {}
            
            # Add description if available (from comment)
            if column.comment:
                field_metadata['description'] = column.comment
            
            # Check for foreign keys
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    if fk.column is not None:
                        # Extract table and column name from foreign key target
                        target_table = fk.column.table.name
                        target_column = fk.column.name
                        foreign_keys[column.name] = (target_table, target_column)
                        # Mark field as foreign_key type
                        table_schema[column.name] = 'foreign_key'
                        break
            
            # Add constraints
            constraints = {}
            
            # Check for nullable
            constraints['nullable'] = column.nullable
            
            # Check for primary key
            if column.primary_key:
                constraints['primary_key'] = True
            
            # Check for unique constraint
            if column.unique:
                constraints['unique'] = True
            
            # Add length constraint for string types
            if hasattr(column.type, 'length') and column.type.length is not None:
                constraints['max_length'] = column.type.length
            
            if constraints:
                field_metadata['constraints'] = constraints
            
            # Add field metadata if any was collected
            if field_metadata:
                metadata[column.name] = field_metadata
        
        return table_schema, metadata, table_description, foreign_keys
    
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
                    # Process as dictionary schema
                    return self._load_dict_schema(schema_dict)
            except Exception as e:
                raise ValueError(f"Error loading JSON schema file {file_path}: {str(e)}")
                
        # YAML file
        elif file_ext in ['.yml', '.yaml']:
            if not YAML_INSTALLED:
                raise ImportError("PyYAML is required for YAML schema support. Install it with 'pip install pyyaml'.")
            
            try:
                with open(file_path, 'r') as f:
                    schema_dict = yaml.safe_load(f)
                    # Process as dictionary schema
                    return self._load_dict_schema(schema_dict)
            except Exception as e:
                raise ValueError(f"Error loading YAML schema file {file_path}: {str(e)}")
        
        # Unsupported file type
        else:
            raise ValueError(f"Unsupported schema file type: {file_ext} for file {file_path}")
            
    def _load_template_class(self, template_class: Type) -> Tuple[Dict, Dict, Optional[str], Dict]:
        """
        Process a template class derived from SydaTemplate or with __template__ attribute.
        
        Args:
            template_class: A class that represents a template
            
        Returns:
            Tuple of (table_schema, metadata, table_description, foreign_keys)
        """
        table_schema = {}
        metadata = {}
        foreign_keys = {}
        
        # Get the class name as table name
        template_name = template_class.__name__
        
        # Get the class docstring as description
        table_description = template_class.__doc__ or f"{template_name} template"
        
        # Extract special metadata attributes
        if hasattr(template_class, '__depends_on__'):
            metadata['__depends_on__'] = template_class.__depends_on__
        
        if hasattr(template_class, '__template_source__'):
            metadata['__template_source__'] = template_class.__template_source__
            # Add as field for template processing
            table_schema['template_source'] = 'text'
            
        if hasattr(template_class, '__input_file_type__'):
            metadata['__input_file_type__'] = template_class.__input_file_type__
            # Add as field for template processing
            table_schema['input_file_type'] = 'text'
            
        if hasattr(template_class, '__output_file_type__'):
            metadata['__output_file_type__'] = template_class.__output_file_type__
            # Add as field for template processing
            table_schema['output_file_type'] = 'text'
        
        # Add __template__ flag to metadata
        metadata['__template__'] = True
        
        # Process class attributes that might be SQLAlchemy Column objects
        for attr_name in dir(template_class):
            # Skip private attributes, methods, and special attributes
            if attr_name.startswith('_') or callable(getattr(template_class, attr_name)) or attr_name in [
                '__template__', '__depends_on__', '__template_source__', 
                '__input_file_type__', '__output_file_type__'
            ]:
                continue
                
            attr = getattr(template_class, attr_name)
            
            # Check if it's a SQLAlchemy Column
            if hasattr(attr, 'type') and hasattr(attr.type, 'python_type'):
                # Get the column type
                column_type = attr.type.__class__.__name__.lower()
                
                # Map SQLAlchemy types to schema types
                if column_type in ('integer', 'biginteger', 'smallinteger'):
                    table_schema[attr_name] = 'integer'
                elif column_type in ('float', 'numeric', 'decimal'):
                    table_schema[attr_name] = 'float'
                elif column_type == 'boolean':
                    table_schema[attr_name] = 'boolean'
                elif column_type == 'date':
                    table_schema[attr_name] = 'date'
                elif column_type == 'datetime':
                    table_schema[attr_name] = 'datetime'
                elif column_type == 'text':
                    table_schema[attr_name] = 'text'
                else:
                    table_schema[attr_name] = 'text'
                
                # Add column metadata
                field_metadata = {}
                
                # Add description if available (from comment)
                if hasattr(attr, 'comment') and attr.comment:
                    field_metadata['description'] = attr.comment
                
                # Check for foreign keys
                if hasattr(attr, 'foreign_keys') and attr.foreign_keys:
                    try:
                        for fk in attr.foreign_keys:
                            # Safely check if we can extract column info
                            try:
                                if hasattr(fk, 'column') and fk.column is not None:
                                    # Extract table and column name from foreign key target
                                    target_table = fk.column.table.name
                                    target_column = fk.column.name
                                    foreign_keys[attr_name] = (target_table, target_column)
                                    # Mark field as foreign_key type
                                    table_schema[attr_name] = 'foreign_key'
                                    print(f"Found template class foreign key: {attr_name} -> {target_table}.{target_column}")
                                    break
                            except Exception as e:
                                # Try to extract from the attribute name - common pattern is field_id -> field.id
                                if attr_name.endswith('_id'):
                                    target_entity = attr_name[:-3]  # Remove '_id' suffix
                                    foreign_keys[attr_name] = (target_entity + 's', 'id')  # Assume plural table name and 'id' field
                                    table_schema[attr_name] = 'foreign_key'
                                    print(f"Inferred foreign key from name: {attr_name} -> {target_entity + 's'}.id")
                    except Exception as e:
                        print(f"Warning: Error processing foreign keys for {attr_name}: {e}")
                        # Handle explicitly declared foreign keys using ForeignKey string notation if available
                        if hasattr(attr, 'target') and hasattr(attr.target, 'name') and '.' in attr.target.name:
                            try:
                                target_parts = attr.target.name.split('.')
                                if len(target_parts) == 2:
                                    target_table, target_column = target_parts
                                    foreign_keys[attr_name] = (target_table, target_column)
                                    table_schema[attr_name] = 'foreign_key'
                                    print(f"Using string-based foreign key: {attr_name} -> {target_table}.{target_column}")
                            except Exception:
                                pass
                
                # Add constraints
                constraints = {}
                
                # Check for nullable
                if hasattr(attr, 'nullable'):
                    constraints['nullable'] = attr.nullable
                
                # Check for primary key
                if hasattr(attr, 'primary_key') and attr.primary_key:
                    constraints['primary_key'] = True
                
                # Check for unique constraint
                if hasattr(attr, 'unique') and attr.unique:
                    constraints['unique'] = True
                
                # Add length constraint for string types
                if column_type in ('string', 'varchar', 'text') and hasattr(attr.type, 'length') and attr.type.length is not None:
                    constraints['max_length'] = attr.type.length
                
                if constraints:
                    field_metadata['constraints'] = constraints
                
                # Add field metadata if any was collected
                if field_metadata:
                    metadata[attr_name] = field_metadata
        
        # Handle foreign key extraction from template class method if available
        if hasattr(template_class, 'get_foreign_keys') and callable(getattr(template_class, 'get_foreign_keys')):
            try:
                template_foreign_keys = template_class.get_foreign_keys()
                for fk_field, fk_info in template_foreign_keys.items():
                    foreign_keys[fk_field] = (fk_info['target_table'], fk_info['target_column'])
                    print(f"Using template-defined foreign key: {fk_field} -> {fk_info['target_table']}.{fk_info['target_column']}")
            except Exception as e:
                print(f"Warning: Error extracting foreign keys from template class: {e}")
        
        return table_schema, metadata, table_description, foreign_keys
