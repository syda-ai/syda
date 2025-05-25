"""
Template handling for unstructured data generation.
"""

import re
import os
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from sqlalchemy import Column, inspect

class SydaTemplate:
    """Base class for document templates with placeholder fields."""
    
    source_path = None  # Path to template file
    
    def __init__(self, **kwargs):
        """Initialize the template with optional overrides."""
        if 'source_path' in kwargs:
            self.source_path = kwargs['source_path']
            
    @classmethod
    def get_source_path(cls):
        """Get the template source path."""
        return cls.source_path
    
    @classmethod
    def get_fields(cls):
        """Get all field definitions from the template class."""
        if not hasattr(cls, '__table__'):
            return {}
            
        fields = {}
        for column in cls.__table__.columns:
            # Skip primary key
            if column.primary_key:
                continue
                
            fields[column.name] = column
                
        return fields
    
    @classmethod
    def get_foreign_keys(cls):
        """Get foreign key relationships from the template class."""
        if not hasattr(cls, '__table__'):
            return {}
            
        foreign_keys = {}
        for column in cls.__table__.columns:
            # Skip columns without foreign keys
            if not column.foreign_keys:
                continue
                
            # Get the first foreign key (there's usually only one per column)
            fk = next(iter(column.foreign_keys))
            target_table = fk.column.table.name
            target_column = fk.column.name
            
            foreign_keys[column.name] = {
                'target_table': target_table,
                'target_column': target_column
            }
                
        return foreign_keys

class TemplateProcessor:
    """Process document templates with placeholders and generate synthetic data."""
    
    def __init__(self, file_processor=None):
        """
        Initialize the template processor.
        
        Args:
            file_processor: Optional file processor for handling different file types.
                           If None, a new UnstructuredDataProcessor will be created.
        """
        # Import here to avoid circular imports
        from .unstructured import UnstructuredDataProcessor
        
        self.file_processor = file_processor or UnstructuredDataProcessor()
        self.placeholder_pattern = re.compile(r'{{\s*([a-zA-Z0-9_]+)\s*}}')
        
    def extract_placeholders(self, text: str) -> Set[str]:
        """
        Extract placeholder field names from text.
        
        Args:
            text: The template text containing placeholders
            
        Returns:
            Set of placeholder field names without the {{ }} delimiters
        """
        return set(self.placeholder_pattern.findall(text))
    
    def get_template_content(self, template_path: str) -> str:
        """
        Extract text content from a template file.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Template text content
            
        Raises:
            ValueError: If content extraction fails
        """
        if not os.path.exists(template_path):
            raise ValueError(f"Template file not found: {template_path}")
            
        result = self.file_processor.process_file(template_path)
        
        if 'error' in result:
            raise ValueError(f"Error processing template file: {result['error']}")
        
        if 'text' not in result:
            raise ValueError(f"Unable to extract text from the template file of type {result['type']}")
        
        return result['text']
    
    def replace_placeholders(self, template_content: str, values: Dict[str, Any]) -> str:
        """
        Replace placeholders in a template with provided values.
        
        Args:
            template_content: Template text with placeholders
            values: Dictionary mapping field names to values
            
        Returns:
            Template with placeholders replaced by values
        """
        result = template_content
        
        for field, value in values.items():
            placeholder = f"{{{{ {field} }}}}"
            result = result.replace(placeholder, str(value))
            
        return result
    
    def create_schema_from_placeholders(self, placeholders: Set[str]) -> Dict[str, str]:
        """
        Create a data schema from extracted placeholders.
        
        Args:
            placeholders: Set of placeholder field names
            
        Returns:
            Dictionary mapping field names to field types
        """
        schema = {}
        
        # Map placeholders to likely field types based on field name patterns
        for field in placeholders:
            field_lower = field.lower()
            
            # Map common field names to appropriate types
            if any(name in field_lower for name in ['name', 'customer', 'client', 'company']):
                schema[field] = 'text'
            elif any(name in field_lower for name in ['email']):
                schema[field] = 'email'
            elif any(name in field_lower for name in ['phone', 'mobile', 'fax']):
                schema[field] = 'phone'
            elif any(name in field_lower for name in ['address', 'street', 'city', 'state', 'zip', 'postal']):
                schema[field] = 'address'
            elif any(name in field_lower for name in ['date', 'time']):
                schema[field] = 'date'
            elif any(name in field_lower for name in ['id', 'customer_id', 'client_id', 'account']):
                schema[field] = 'id'
            elif any(name in field_lower for name in ['amount', 'price', 'cost', 'fee', 'total']):
                schema[field] = 'number'
            else:
                # Default to text for unknown field types
                schema[field] = 'text'
                
        return schema
