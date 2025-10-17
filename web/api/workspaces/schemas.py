"""
Pydantic schemas for Workspace API
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class WorkspaceBase(BaseModel):
    """Base workspace schema"""
    name: str = Field(..., min_length=1, max_length=255, description="Workspace name")
    description: Optional[str] = Field(None, description="Workspace description")


class WorkspaceCreate(WorkspaceBase):
    """Schema for creating a workspace"""
    pass


class WorkspaceUpdate(BaseModel):
    """Schema for updating a workspace"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class WorkspaceResponse(WorkspaceBase):
    """Schema for workspace response"""
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class WorkspaceListResponse(BaseModel):
    """Schema for listing workspaces"""
    workspaces: list[WorkspaceResponse]
    total: int

