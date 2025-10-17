"""
API routes for Provider management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from database import get_db
from . import schemas, service
import uuid

router = APIRouter()


@router.get("/", response_model=schemas.ProviderListResponse)
def list_providers(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: Session = Depends(get_db)
):
    """
    List all AI providers
    
    Returns a list of all configured AI providers with masked API keys.
    """
    try:
        return service.get_all_providers(db, skip=skip, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve providers: {str(e)}"
        )


@router.get("/{provider_id}", response_model=schemas.ProviderResponse)
def get_provider(
    provider_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific provider by ID
    
    Returns detailed information about a single provider with masked API key.
    """
    try:
        return service.get_provider_by_id(db, provider_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve provider: {str(e)}"
        )


@router.post("/", response_model=schemas.ProviderResponse, status_code=status.HTTP_201_CREATED)
def create_provider(
    provider: schemas.ProviderCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new AI provider configuration
    
    Creates a new provider with encrypted API key storage.
    """
    try:
        return service.create_provider_secure(db, provider)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create provider: {str(e)}"
        )


@router.put("/{provider_id}", response_model=schemas.ProviderResponse)
def update_provider(
    provider_id: uuid.UUID,
    provider: schemas.ProviderUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an AI provider configuration
    
    Updates provider settings. API key will be re-encrypted if provided.
    """
    try:
        return service.update_provider_secure(db, provider_id, provider)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update provider: {str(e)}"
        )


@router.delete("/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_provider(
    provider_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Delete an AI provider
    
    Permanently removes a provider configuration.
    """
    try:
        service.delete_provider_secure(db, provider_id)
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete provider: {str(e)}"
        )


@router.post("/{provider_id}/test", response_model=schemas.ProviderTestResponse)
def test_provider(
    provider_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Test provider API connection
    
    Tests the provider's API connection and updates its status.
    """
    try:
        return service.test_provider_connection(db, provider_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test connection: {str(e)}"
        )

