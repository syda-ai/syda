"""
CRUD operations for Provider model
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from models import Provider
from . import schemas
from typing import List, Optional
import uuid
from datetime import datetime


def get_provider(db: Session, provider_id: uuid.UUID) -> Optional[Provider]:
    """
    Get a single provider by ID
    
    Args:
        db: Database session
        provider_id: Provider UUID
        
    Returns:
        Provider object or None
    """
    return db.query(Provider).filter(Provider.id == provider_id).first()


def get_provider_by_key(db: Session, key: str) -> Optional[Provider]:
    """
    Get provider by key (anthropic, openai, etc.)
    
    Args:
        db: Database session
        key: Provider key
        
    Returns:
        Provider object or None
    """
    return db.query(Provider).filter(Provider.key == key).first()


def get_providers(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None
) -> List[Provider]:
    """
    Get all providers with optional filtering
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        status: Optional status filter
        
    Returns:
        List of Provider objects
    """
    query = db.query(Provider)
    
    if status:
        query = query.filter(Provider.status == status)
    
    return query.offset(skip).limit(limit).all()


def get_providers_count(db: Session, status: Optional[str] = None) -> int:
    """
    Get total count of providers
    
    Args:
        db: Database session
        status: Optional status filter
        
    Returns:
        Count of providers
    """
    query = db.query(Provider)
    
    if status:
        query = query.filter(Provider.status == status)
    
    return query.count()


def create_provider(db: Session, provider: schemas.ProviderCreate) -> Provider:
    """
    Create a new provider
    
    Args:
        db: Database session
        provider: Provider creation schema
        
    Returns:
        Created Provider object
    """
    db_provider = Provider(
        name=provider.name,
        key=provider.key,
        api_key=provider.api_key,  # Should be encrypted before calling this
        extra_kwargs=provider.extra_kwargs or {},
        status="not-configured"
    )
    db.add(db_provider)
    db.commit()
    db.refresh(db_provider)
    return db_provider


def update_provider(
    db: Session, 
    provider_id: uuid.UUID, 
    provider_update: schemas.ProviderUpdate
) -> Optional[Provider]:
    """
    Update a provider
    
    Args:
        db: Database session
        provider_id: Provider UUID
        provider_update: Update schema
        
    Returns:
        Updated Provider object or None
    """
    db_provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not db_provider:
        return None
    
    # Update only provided fields
    update_data = provider_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_provider, field, value)
    
    db.commit()
    db.refresh(db_provider)
    return db_provider


def delete_provider(db: Session, provider_id: uuid.UUID) -> bool:
    """
    Delete a provider
    
    Args:
        db: Database session
        provider_id: Provider UUID
        
    Returns:
        True if deleted, False if not found
    """
    db_provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not db_provider:
        return False
    
    db.delete(db_provider)
    db.commit()
    return True


def update_provider_status(
    db: Session, 
    provider_id: uuid.UUID, 
    status: str, 
    last_tested: Optional[datetime] = None
) -> Optional[Provider]:
    """
    Update provider status after testing
    
    Args:
        db: Database session
        provider_id: Provider UUID
        status: New status
        last_tested: Optional test timestamp
        
    Returns:
        Updated Provider object or None
    """
    db_provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not db_provider:
        return None
    
    db_provider.status = status
    if last_tested:
        db_provider.last_tested = last_tested
    
    db.commit()
    db.refresh(db_provider)
    return db_provider

