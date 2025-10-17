"""
SQLAlchemy ORM models for the Syda API
"""
from sqlalchemy import Column, String, Integer, JSON, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from database import Base


class Workspace(Base):
    """Projects/Workspaces to organize schemas and jobs"""
    __tablename__ = "workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    schemas = relationship("Schema", back_populates="workspace", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="workspace", cascade="all, delete-orphan")


class Provider(Base):
    """AI Provider configurations"""
    __tablename__ = "providers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    key = Column(String(50), unique=True, nullable=False, index=True)  # anthropic, openai, etc.
    api_key = Column(Text, nullable=False)  # Encrypted
    extra_kwargs = Column(JSON, default={}, nullable=False)
    status = Column(String(20), default="not-configured", nullable=False)  # connected, testing, error, not-configured
    last_tested = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Schema(Base):
    """Data schema definitions (structured/template)"""
    __tablename__ = "schemas"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    type = Column(String(20), nullable=False)  # structured, template
    definition = Column(JSON, nullable=False)  # Full schema definition
    dependencies = Column(JSON, default=[], nullable=False)  # List of dependent schema IDs/names
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    workspace = relationship("Workspace", back_populates="schemas")
    jobs = relationship("Job", back_populates="schema")


class Job(Base):
    """Data generation jobs"""
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    schema_id = Column(UUID(as_uuid=True), ForeignKey("schemas.id", ondelete="CASCADE"), nullable=False)
    provider_key = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    sample_size = Column(Integer, nullable=False)
    model_config = Column(JSON, default={}, nullable=False)  # temperature, max_tokens, etc.
    status = Column(String(20), default="pending", nullable=False)  # pending, running, completed, failed
    progress = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="jobs")
    schema = relationship("Schema", back_populates="jobs")
    result = relationship("Result", back_populates="job", uselist=False, cascade="all, delete-orphan")


class Result(Base):
    """Job results and file storage"""
    __tablename__ = "results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)  # Path to generated file
    file_format = Column(String(10), nullable=False)  # csv, json, parquet
    file_size = Column(Integer, nullable=True)  # bytes
    row_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    job = relationship("Job", back_populates="result")


class Setting(Base):
    """User settings and preferences"""
    __tablename__ = "settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    theme = Column(String(10), default="dark", nullable=False)  # dark, light
    default_provider = Column(String(50), nullable=True)
    default_workspace_id = Column(UUID(as_uuid=True), nullable=True)
    preferences = Column(JSON, default={}, nullable=False)  # Additional preferences
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

