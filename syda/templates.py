"""
Template handling for unstructured data generation.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from sqlalchemy import Column, inspect
import jinja2

class SydaTemplate:
    """Base class for document templates with placeholder fields."""
    
    # Legacy attribute
    source_path = None  # Path to template file
    
    # New template configuration attributes using double-underscore style
    __template__ = False  # Flag to indicate this is a template class
    __template_source__ = None  # Path to template file
    __input_file_type__ = None  # Input file type (html, txt, etc.)
    __output_file_type__ = None  # Output file type (pdf, html, etc.)
    __depends_on__ = []  # List of model names this template depends on
    
    def __init__(self, **kwargs):
        """Initialize the template with optional overrides."""
        if 'source_path' in kwargs:
            self.source_path = kwargs['source_path']
        
        # Copy all provided kwargs to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    @classmethod
    def get_source_path(cls):
        """Get the template source path."""
        # First check for new attribute, fall back to legacy
        return cls.__template_source__ if hasattr(cls, '__template_source__') else cls.source_path
    
    @classmethod
    def get_fields(cls):
        """Get all field definitions from the template class."""
        fields = {}
        
        # If SQLAlchemy model, get columns from table
        if hasattr(cls, '__table__'):
            for column in cls.__table__.columns:
                # Skip primary key
                if column.primary_key:
                    continue
                    
                fields[column.name] = column
        
        # Otherwise get all non-special attributes directly from class
        else:
            for attr_name in dir(cls):
                # Skip special attributes, methods, and private attributes
                if (attr_name.startswith('__') and attr_name.endswith('__')) or \
                   callable(getattr(cls, attr_name)) or \
                   attr_name.startswith('_'):
                    continue
                    
                fields[attr_name] = getattr(cls, attr_name)
                
        return fields
    
    @classmethod
    def get_foreign_keys(cls):
        """Get foreign key relationships from the template class."""
        foreign_keys = {}
        
        # Case 1: Check for explicitly defined __foreign_keys__ dictionary
        if hasattr(cls, '__foreign_keys__'):
            for fk_col, fk_ref in cls.__foreign_keys__.items():
                if isinstance(fk_ref, (list, tuple)) and len(fk_ref) == 2:
                    target_table, target_column = fk_ref
                    foreign_keys[fk_col] = {
                        'target_table': target_table,
                        'target_column': target_column
                    }
        
        # Case 2: Check for SQLAlchemy Column objects with ForeignKey constraints
        for attr_name in dir(cls):
            # Skip special attributes, methods, and private attributes
            if (attr_name.startswith('__') and attr_name.endswith('__')) or \
               callable(getattr(cls, attr_name)) or \
               attr_name.startswith('_'):
                continue
            
            attr_value = getattr(cls, attr_name)
            
            # Check if the attribute is a Column with ForeignKey
            if isinstance(attr_value, Column) and hasattr(attr_value, 'foreign_keys') and attr_value.foreign_keys:
                # Extract the foreign key reference
                for fk in attr_value.foreign_keys:
                    if hasattr(fk, '_colspec') and fk._colspec is not None:
                        # Parse the colspec which is in format 'table.column'
                        parts = fk._colspec.split('.')
                        if len(parts) == 2:
                            target_table, target_column = parts
                            foreign_keys[attr_name] = {
                                'target_table': target_table,
                                'target_column': target_column
                            }
                            break
        
        # Case 3: If the class has a __table__, also check column foreign_keys from there
        if hasattr(cls, '__table__'):
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
        
    def render_jinja2_template(self, template_path: str, data: Dict[str, Any]) -> str:
        """
        Render a Jinja2 template with data.
        
        Args:
            template_path: Path to the template file
            data: Dictionary of data to fill the template
            
        Returns:
            Rendered template content
        """
        template_dir = os.path.dirname(os.path.abspath(template_path))
        template_file = os.path.basename(template_path)
        
        # Set up Jinja2 environment
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        try:
            # Load and render the template
            template = env.get_template(template_file)
            return template.render(**data)
        except Exception as e:
            raise ValueError(f"Error rendering Jinja2 template: {str(e)}")
    
    def process_template_with_data(self, template_path: str, data: Dict[str, Any], 
                               output_path: Optional[str] = None,
                               input_file_type: Optional[str] = None,
                               output_file_type: Optional[str] = None) -> str:
        """
        Process a template with data and generate output file.
        
        Args:
            template_path: Path to the template file
            data: Dictionary of data to fill the template
            output_path: Path to save the output file
            input_file_type: Type of input file (html, rtf, txt)
            output_file_type: Type of output file (pdf, html, rtf, txt)
            
        Returns:
            Path to the generated file or content string
        """
        # Determine file types if not provided
        if not input_file_type:
            _, ext = os.path.splitext(template_path)
            input_file_type = ext.lstrip('.').lower() if ext else 'txt'
            
        if not output_file_type:
            output_file_type = input_file_type
        
        # For HTML templates with Jinja2 syntax, use Jinja renderer
        if input_file_type and input_file_type.lower() in ['html', 'htm'] and os.path.exists(template_path):
            try:
                # Try Jinja2 rendering first
                filled_content = self.render_jinja2_template(template_path, data)
            except Exception as e:
                print(f"Jinja2 rendering failed, falling back to placeholder replacement: {str(e)}")
                # Fall back to regular placeholder replacement
                template_content = self.get_template_content(template_path)
                filled_content = self.replace_placeholders(template_content, data)
        else:
            # Get template content
            try:
                template_content = self.get_template_content(template_path)
            except ValueError as e:
                # If we can't get content via the processor, try direct file reading
                if os.path.exists(template_path):
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                else:
                    raise ValueError(f"Unable to read template: {str(e)}")
            
            # Replace placeholders with data
            filled_content = self.replace_placeholders(template_content, data)
        
        # If no output path, return the content
        if not output_path:
            return filled_content
        
        # Process based on input/output types
        if input_file_type.lower() in ['html', 'htm']:
            return self._process_html_template(filled_content, output_path, output_file_type)
        elif input_file_type.lower() == 'rtf':
            return self._process_rtf_template(filled_content, output_path, output_file_type)
        else:
            # Default to text processing
            return self._process_text_template(filled_content, output_path, output_file_type)

    def _process_html_template(self, content: str, output_path: str, output_file_type: str) -> str:
        """Process HTML template content and convert to output format."""
        # Save as HTML
        html_path = output_path
        if output_file_type != 'html':
            html_path = f"{output_path}.html"
            
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Convert to PDF if requested
        if output_file_type == 'pdf':
            try:
                from weasyprint import HTML
                pdf_path = output_path
                if not pdf_path.endswith('.pdf'):
                    pdf_path = f"{os.path.splitext(output_path)[0]}.pdf"
                    
                HTML(string=content).write_pdf(pdf_path)
                
                # Remove intermediate HTML if it was created
                if html_path != output_path and os.path.exists(html_path):
                    os.remove(html_path)
                    
                return pdf_path
            except ImportError:
                print("WeasyPrint not installed. Saving as HTML only.")
                return html_path
        
        return html_path

    def _process_rtf_template(self, content: str, output_path: str, output_file_type: str) -> str:
        """Process RTF template content and convert to output format."""
        # Save as RTF
        rtf_path = output_path
        if not rtf_path.endswith('.rtf'):
            rtf_path = f"{os.path.splitext(output_path)[0]}.rtf"
            
        with open(rtf_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Convert to PDF if requested
        if output_file_type == 'pdf':
            try:
                # Try using LibreOffice for conversion
                import subprocess
                pdf_path = f"{os.path.splitext(output_path)[0]}.pdf"
                output_dir = os.path.dirname(output_path)
                
                result = subprocess.run([
                    'libreoffice', '--headless', '--convert-to', 'pdf',
                    '--outdir', output_dir,
                    rtf_path
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"LibreOffice conversion failed: {result.stderr}")
                    
                # Remove intermediate RTF if different from output path
                if rtf_path != output_path and os.path.exists(rtf_path):
                    os.remove(rtf_path)
                    
                return pdf_path
            except Exception as e:
                print(f"PDF conversion failed: {str(e)}. Saving as RTF only.")
                return rtf_path
        
        return rtf_path

    def _process_text_template(self, content: str, output_path: str, output_file_type: str) -> str:
        """Process text template content and save to output format."""
        # Handle different output types for text templates
        if output_file_type == 'pdf':
            # For PDF, we'll first save as HTML and then convert
            html_content = f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n</head>\n<body>\n<pre>{content}</pre>\n</body>\n</html>"
            html_path = f"{os.path.splitext(output_path)[0]}.html"
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            try:
                from weasyprint import HTML
                pdf_path = f"{os.path.splitext(output_path)[0]}.pdf"
                HTML(string=html_content).write_pdf(pdf_path)
                
                # Remove intermediate HTML
                if os.path.exists(html_path):
                    os.remove(html_path)
                    
                return pdf_path
            except ImportError:
                print("WeasyPrint not installed. Saving as text only.")
                # Fall through to text saving
        
        # Save as text (default)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return output_path
        
    def process_template_dataframes(self, template_dataframes, output_dir=None):
        """
        Process multiple template dataframes, generating documents for each row.
        
        Args:
            template_dataframes: Dictionary mapping schema names to dataframes with template data
            output_dir: Base directory for output files
            
        Returns:
            Dictionary mapping schema names to lists of generated document paths
            
        Raises:
            ValueError: If processing fails or output_dir is not provided
        """
        if not output_dir:
            raise ValueError("Output directory must be specified for template processing")
            
        results = {}
        
        for schema_name, df in template_dataframes.items():
            # Skip if not a template dataframe
            if df is None or 'template_source' not in df.columns:
                continue
                
            print(f"Processing {schema_name} templates...")
            
            # Create output directory for this schema
            template_output_dir = os.path.join(output_dir, schema_name)
            os.makedirs(template_output_dir, exist_ok=True)
            
            # Process each row in the dataframe
            documents_generated = 0
            schema_results = []
            
            for idx, row in df.iterrows():
                template_path = row.get('template_source')
                input_file_type = row.get('input_file_type', '').lower()
                output_file_type = row.get('output_file_type', '').lower()
                
                # Skip if missing required fields
                if not template_path or not os.path.exists(template_path) or not input_file_type or not output_file_type:
                    print(f"Warning: Invalid template configuration for {schema_name} row {idx}")
                    continue
                    
                # Output path for this document
                output_path = os.path.join(template_output_dir, f"document_{idx+1}.{output_file_type}")
                
                try:
                    # Process the template with data
                    self.process_template_with_data(
                        template_path=template_path,
                        data=row.to_dict(),
                        output_path=output_path,
                        input_file_type=input_file_type,
                        output_file_type=output_file_type
                    )
                    documents_generated += 1
                    schema_results.append(output_path)
                    print(f"✓ Successfully generated: {output_path}")
                except Exception as e:
                    print(f"Error generating document for {schema_name} row {idx}: {str(e)}")
            
            print(f"Generated {documents_generated} documents for {schema_name}")
            results[schema_name] = schema_results
            
        return results
