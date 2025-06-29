"""
Schema definitions for the syda library.
Contains Pydantic models used for data validation and configuration.
"""

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Dict, Optional, Any, Literal, List, Union, Set
import re


class ProxyConfig(BaseModel):
    """
    Configuration for API proxy settings, commonly used in enterprise environments.
    """
    
    # Base URL for the proxy service
    base_url: Optional[str] = Field(None, description="Base URL for the proxy service (e.g. 'https://ai-proxy.company.com/v1')")
    
    # Additional headers to include in requests
    headers: Optional[Dict[str, str]] = Field(None, description="Additional HTTP headers to include in requests to the proxy")
    
    # Additional query parameters to include in URL
    params: Optional[Dict[str, Any]] = Field(None, description="Additional query parameters to include in the URL")
    
    # Path modification (some proxies require a different path structure)
    path_format: Optional[str] = Field(None, description="Optional format string for proxy path, e.g. '/proxy/{provider}/{endpoint}'")
    
    def get_proxy_kwargs(self) -> Dict[str, Any]:
        """Get proxy-specific kwargs for API client initialization."""
        kwargs = {}
        
        # Handle base_url and query parameters
        if self.base_url:
            base_url = self.base_url
            
            # Add query parameters to the base URL if provided
            if self.params:
                # Convert all param values to strings
                params = {k: str(v) for k, v in self.params.items()}
                
                # Create the query string
                from urllib.parse import urlencode
                query_string = urlencode(params)
                
                # Append to base_url with ? or & depending on whether URL already has params
                if "?" in base_url:
                    base_url += "&" + query_string
                else:
                    base_url += "?" + query_string
                    
            kwargs["base_url"] = base_url
            
        # Handle custom headers
        if self.headers:
            kwargs["default_headers"] = self.headers
            
        return kwargs

class ModelConfig(BaseModel):
    """
    Configuration for AI model settings used by the SyntheticDataGenerator.
    
    This class provides a structured way to define the model and its parameters,
    supporting both OpenAI and Anthropic models with provider-specific settings.
    """
    
    # Model provider and name
    provider: Literal["openai", "anthropic"] = "anthropic"
    model_name: str = "claude-3-5-haiku-20241022"
    
    
    temperature: float = Field(None, ge=0.0, le=1.0, description="Controls randomness: 0.0 is deterministic, higher values are more random")
    max_tokens: int = Field(None, description="Maximum number of tokens to generate. Larger values allow for more complete responses.")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    
    # Streaming parameters
    stream: Optional[bool] = Field(
        False, 
        description="""
        Enable streaming responses for certain models.
        If not provided or set to False, 
        streaming for models will be enabled if sample size is >50 or for certain anthropic models.
        """
    )

    # OpenAI specific parameters
    seed: Optional[int] = Field(None, description="Random seed for reproducibility (OpenAI only)")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Format for responses (OpenAI only)")
    max_completion_tokens: Optional[int] = Field(None, description="Maximum completion tokens (OpenAI only)")
    
    # Anthropic specific parameters
    top_k: Optional[int] = Field(None, description="Top K sampling parameter (Anthropic only)")
    max_tokens_to_sample: Optional[int] = Field(None, description="Maximum tokens to generate (Anthropic only)")
    
    # Proxy configuration
    proxy: Optional[ProxyConfig] = Field(None, description="Optional proxy configuration for API requests")
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model-specific kwargs for API calls."""
        # Start with common parameters
        kwargs = {}
        
        # Always include the model name, which is required for OpenAI and used for other providers
        kwargs["model"] = self.model_name
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        
        # Add provider-specific parameters
        if self.provider == "openai":
            if self.top_p:
                kwargs["top_p"] = self.top_p
            if self.seed:
                kwargs["seed"] = self.seed
            if self.response_format:
                kwargs["response_format"] = self.response_format
            if self.max_completion_tokens:
                kwargs["max_completion_tokens"] = self.max_completion_tokens
    
               
        
        elif self.provider == "anthropic":
            # The updated Anthropic API via instructor now uses the same parameter names as OpenAI
            # for better compatibility across providers
            
            # Override max_tokens with max_tokens_to_sample if provided (for backward compatibility)
            if self.max_tokens_to_sample:
                kwargs["max_tokens"] = self.max_tokens_to_sample
                
            if self.top_p:
                kwargs["top_p"] = self.top_p
                
            if self.top_k:
                # This might not be supported in the current instructor integration
                # but we'll keep it for future compatibility
                kwargs["top_k"] = self.top_k
                
            # Ensure model name is set correctly for Anthropic
            kwargs["model"] = self.model_name
        
        return kwargs


# Schema Validation Models

class FieldConstraint(BaseModel):
    """Base model for field constraints"""
    nullable: Optional[bool] = None
    unique: Optional[bool] = None
    primary_key: Optional[bool] = None


class NumericConstraint(FieldConstraint):
    """Constraints for numeric fields"""
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    decimals: Optional[int] = None
    enum: Optional[List[Union[int, float]]] = None
    
    @model_validator(mode="after")
    def validate_min_max(self):
        if self.min is not None and self.max is not None:
            if self.min > self.max:
                raise ValueError(f"min value ({self.min}) cannot be greater than max value ({self.max})")
        return self


class StringConstraint(FieldConstraint):
    """Constraints for string fields"""
    length: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    format: Optional[str] = None
    enum: Optional[List[str]] = None
    
    @model_validator(mode="after")
    def validate_length_constraints(self):
        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError(f"min_length ({self.min_length}) cannot be greater than max_length ({self.max_length})")
        
        if self.length is not None and (self.min_length is not None or self.max_length is not None):
            raise ValueError("Cannot specify both 'length' and 'min_length'/'max_length'")
        
        return self
    
    @field_validator("pattern")
    def validate_pattern(cls, v):
        if v is not None:
            try:
                re.compile(v)
            except re.error:
                raise ValueError(f"Invalid regex pattern: {v}")
        return v
    
    @field_validator("enum")
    def validate_enum(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("Enum list cannot be empty")
        return v


class DateConstraint(FieldConstraint):
    """Constraints for date fields"""
    min: Optional[str] = None
    max: Optional[str] = None
    format: Optional[str] = None


class ArrayConstraint(FieldConstraint):
    """Constraints for array fields"""
    items: Optional[Dict[str, Any]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    @model_validator(mode="after")
    def validate_items(self):
        if self.items is not None:
            if not isinstance(self.items, dict):
                raise ValueError("'items' must be a dictionary")
            if 'type' not in self.items:
                raise ValueError("'items' must contain a 'type' field")
        return self


class ForeignKeyConstraint(FieldConstraint):
    """Constraints for foreign key fields"""
    references: Optional[Union[str, List[str]]] = None


class FieldMetadata(BaseModel):
    """Metadata for a field in a schema"""
    description: Optional[str] = None
    constraints: Optional[Union[
        NumericConstraint,
        StringConstraint,
        DateConstraint,
        ArrayConstraint,
        ForeignKeyConstraint,
        Dict[str, Any]
    ]] = None


class SchemaField(BaseModel):
    """A field in a schema with its type and metadata"""
    type: str
    description: Optional[str] = None
    constraints: Optional[Union[
        NumericConstraint,
        StringConstraint,
        DateConstraint,
        ArrayConstraint,
        ForeignKeyConstraint,
        Dict[str, Any]
    ]] = None


class ForeignKeyDefinition(BaseModel):
    """Definition of a foreign key reference"""
    table: str
    column: str


class SchemaTemplate(BaseModel):
    """Template-related fields in a schema"""
    template: Optional[Union[bool, str]] = Field(None, alias="__template__")
    template_source: Optional[str] = Field(None, alias="__template_source__")
    input_file_type: Optional[str] = Field(None, alias="__input_file_type__")
    output_file_type: Optional[str] = Field(None, alias="__output_file_type__")
    
    @model_validator(mode="after")
    def validate_template(self):
        if self.template is not None and not self.template_source:
            raise ValueError("Template schema missing required field '__template_source__'")
        return self

class Schema(BaseModel):
    """A complete schema definition"""
    # Special fields
    description: Optional[str] = Field(None, alias="__description__")
    table_description: Optional[str] = Field(None, alias="__table_description__")
    foreign_keys: Optional[Dict[str, Union[str, List[str], ForeignKeyDefinition]]] = Field(None, alias="__foreign_keys__")
    
    # Template fields
    template: Optional[Union[bool, str]] = Field(None, alias="__template__")
    template_source: Optional[str] = Field(None, alias="__template_source__")
    input_file_type: Optional[str] = Field(None, alias="__input_file_type__")
    output_file_type: Optional[str] = Field(None, alias="__output_file_type__")
    
    # Additional fields will be validated with the field validator
    fields: Dict[str, Union[str, Dict, SchemaField]] = Field(default_factory=dict)
    
    # Valid field types
    VALID_TYPES: Set[str] = {
        # Core types
        'text', 'string', 'integer', 'int', 'float', 'number', 
        'boolean', 'bool', 'date', 'datetime', 'array', 'object',
        'foreign_key',
        
        # Extended types
        'email', 'phone', 'address', 'url'
    }
    
    @model_validator(mode="before")
    @classmethod
    def extract_fields(cls, data):
        """Extract fields from the schema"""
        if not isinstance(data, dict):
            return data
        
        fields = {}
        for key, value in list(data.items()):
            # Skip special fields
            if key.startswith('__') and key.endswith('__'):
                continue
                
            # Move field to fields dict
            fields[key] = value
            data.pop(key)
            
        data['fields'] = fields
        return data
    
    @model_validator(mode="after")
    def validate_fields(self):
        """Validate field types and constraints"""
        for field_name, field_def in self.fields.items():
            # Handle string field type
            if isinstance(field_def, str):
                if field_def not in self.VALID_TYPES:
                    raise ValueError(f"Field '{field_name}' has invalid type: '{field_def}'")
            
            # Handle dict/SchemaField field type
            elif isinstance(field_def, dict):
                field_type = field_def.get('type')
                if not field_type:
                    raise ValueError(f"Field '{field_name}' is missing 'type' specification")
                if field_type not in self.VALID_TYPES:
                    raise ValueError(f"Field '{field_name}' has invalid type: '{field_type}'")
            
        # Validate template fields
        if self.template is not None and not self.template_source:
            raise ValueError("Template schema missing required field '__template_source__'")
            
        # Validate foreign keys
        if self.foreign_keys:
            for fk_field, reference in self.foreign_keys.items():
                if fk_field not in self.fields and not fk_field.startswith('__'):
                    raise ValueError(f"Foreign key '{fk_field}' refers to non-existent field")
        
        return self


def validate_schema(schema_dict: Dict) -> Dict:
    """
    Validate a schema dictionary using Pydantic models
    
    Args:
        schema_dict: Dictionary containing the schema definition
        
    Returns:
        The validated schema dictionary (original structure preserved)
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Create a copy of the schema dict to not modify the original
        validation_copy = dict(schema_dict)
        
        # Pre-process the template field if it exists to ensure it's compatible
        if "__template__" in validation_copy and isinstance(validation_copy["__template__"], bool):
            # If it's a boolean, we already made the model accept it, so that's fine
            pass
            
        # Validate with pydantic
        schema = Schema.model_validate(validation_copy)
        
        # Important: Return the original schema_dict, not the validated one
        # This preserves the original structure
        return schema_dict
    except Exception as e:
        error_msg = f"Schema validation failed: {str(e)}\n"
        # Add more debug info about which fields seem to be causing problems
        if "__template__" in schema_dict:
            error_msg += f"__template__ field type: {type(schema_dict['__template__']).__name__}, value: {schema_dict['__template__']}\n"
        raise ValueError(error_msg)