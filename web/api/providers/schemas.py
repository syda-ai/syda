"""
Pydantic schemas for Provider API
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from datetime import datetime
import uuid
from utils.validators import validate_extra_kwargs, validate_provider_key


class ProviderBase(BaseModel):
    """Base provider schema"""
    name: str = Field(..., min_length=1, max_length=100, description="Provider display name")
    key: str = Field(..., min_length=1, max_length=50, description="Provider key (anthropic, openai, etc.)")
    extra_kwargs: Dict[str, str] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        return validate_provider_key(v)
    
    @field_validator('extra_kwargs')
    @classmethod
    def validate_kwargs(cls, v):
        return validate_extra_kwargs(v)


class ProviderCreate(ProviderBase):
    """Schema for creating a new provider"""
    api_key: str = Field(..., min_length=1, description="Provider API key")


class ProviderUpdate(BaseModel):
    """Schema for updating a provider"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    api_key: Optional[str] = Field(None, min_length=1)
    extra_kwargs: Optional[Dict[str, str]] = None
    status: Optional[str] = Field(None, pattern="^(connected|testing|error|not-configured)$")
    
    @field_validator('extra_kwargs')
    @classmethod
    def validate_kwargs(cls, v):
        if v is not None:
            return validate_extra_kwargs(v)
        return v


class ProviderResponse(ProviderBase):
    """Schema for provider response"""
    id: uuid.UUID
    api_key: str  # Will be masked
    status: str = Field(..., description="Connection status")
    last_tested: Optional[datetime] = Field(None, description="Last connection test timestamp")
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ProviderListResponse(BaseModel):
    """Schema for listing providers"""
    providers: list[ProviderResponse]
    total: int


class ProviderTestResponse(BaseModel):
    """Schema for connection test response"""
    status: str = Field(..., description="Test result status")
    message: str = Field(..., description="Test result message")
    provider: ProviderResponse

