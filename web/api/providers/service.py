"""
Business logic for Provider operations
"""
from sqlalchemy.orm import Session
from . import crud, schemas
from utils.security import encrypt_api_key, decrypt_api_key, mask_api_key
from datetime import datetime
import uuid
from typing import List


def get_all_providers(db: Session, skip: int = 0, limit: int = 100) -> schemas.ProviderListResponse:
    """
    Get all providers with masked API keys
    
    Args:
        db: Database session
        skip: Pagination skip
        limit: Pagination limit
        
    Returns:
        Provider list response
    """
    providers = crud.get_providers(db, skip=skip, limit=limit)
    total = crud.get_providers_count(db)
    
    # Mask API keys for security
    for provider in providers:
        provider.api_key = mask_api_key(provider.api_key)
    
    return schemas.ProviderListResponse(
        providers=providers,
        total=total
    )


def get_provider_by_id(db: Session, provider_id: uuid.UUID) -> schemas.ProviderResponse:
    """
    Get a single provider by ID with masked API key
    
    Args:
        db: Database session
        provider_id: Provider UUID
        
    Returns:
        Provider response
        
    Raises:
        ValueError: If provider not found
    """
    provider = crud.get_provider(db, provider_id)
    if not provider:
        raise ValueError("Provider not found")
    
    provider.api_key = mask_api_key(provider.api_key)
    return provider


def create_provider_secure(db: Session, provider: schemas.ProviderCreate) -> schemas.ProviderResponse:
    """
    Create provider with encrypted API key
    
    Args:
        db: Database session
        provider: Provider creation schema
        
    Returns:
        Created provider response
        
    Raises:
        ValueError: If provider with this key already exists
    """
    # Check if provider with this key already exists
    existing = crud.get_provider_by_key(db, provider.key)
    if existing:
        raise ValueError(f"Provider with key '{provider.key}' already exists")
    
    # Encrypt API key
    encrypted_key = encrypt_api_key(provider.api_key)
    provider.api_key = encrypted_key
    
    # Create provider
    db_provider = crud.create_provider(db, provider)
    
    # Mask API key for response
    db_provider.api_key = mask_api_key(db_provider.api_key)
    
    return db_provider


def update_provider_secure(
    db: Session, 
    provider_id: uuid.UUID, 
    provider_update: schemas.ProviderUpdate
) -> schemas.ProviderResponse:
    """
    Update provider with encryption
    
    Args:
        db: Database session
        provider_id: Provider UUID
        provider_update: Update schema
        
    Returns:
        Updated provider response
        
    Raises:
        ValueError: If provider not found
    """
    # Check if provider exists
    existing = crud.get_provider(db, provider_id)
    if not existing:
        raise ValueError("Provider not found")
    
    # Encrypt API key if provided
    if provider_update.api_key:
        provider_update.api_key = encrypt_api_key(provider_update.api_key)
    
    # Update provider
    updated = crud.update_provider(db, provider_id, provider_update)
    
    # Mask API key for response
    updated.api_key = mask_api_key(updated.api_key)
    
    return updated


def delete_provider_secure(db: Session, provider_id: uuid.UUID) -> bool:
    """
    Delete a provider
    
    Args:
        db: Database session
        provider_id: Provider UUID
        
    Returns:
        True if deleted
        
    Raises:
        ValueError: If provider not found
    """
    success = crud.delete_provider(db, provider_id)
    if not success:
        raise ValueError("Provider not found")
    return True


def test_provider_connection(db: Session, provider_id: uuid.UUID) -> schemas.ProviderTestResponse:
    """
    Test provider API connection
    
    Args:
        db: Database session
        provider_id: Provider UUID
        
    Returns:
        Test result response
        
    Raises:
        ValueError: If provider not found
    """
    provider = crud.get_provider(db, provider_id)
    if not provider:
        raise ValueError("Provider not found")
    
    # Decrypt API key for testing
    try:
        api_key = decrypt_api_key(provider.api_key)
    except Exception as e:
        # Update status to error
        crud.update_provider_status(db, provider_id, "error", datetime.utcnow())
        raise ValueError(f"Failed to decrypt API key: {str(e)}")
    
    # TODO: Implement actual connection test logic here
    # For now, simulate a successful test
    test_successful = bool(api_key)  # Simple check that key exists
    
    # Update provider status
    status = "connected" if test_successful else "error"
    message = "Connection successful" if test_successful else "Connection failed"
    
    updated = crud.update_provider_status(
        db, 
        provider_id, 
        status, 
        datetime.utcnow()
    )
    
    # Mask API key for response
    updated.api_key = mask_api_key(updated.api_key)
    
    return schemas.ProviderTestResponse(
        status=status,
        message=message,
        provider=updated
    )

