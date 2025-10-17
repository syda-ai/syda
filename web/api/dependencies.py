"""
Common dependencies for the API
"""
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from typing import Optional


# Database dependency (re-exported for convenience)
def get_database() -> Session:
    """Get database session"""
    return Depends(get_db)


# Example: Authentication dependency (for future use)
async def get_current_user(
    # token: str = Depends(oauth2_scheme),
    # db: Session = Depends(get_db)
):
    """
    Get current authenticated user
    TODO: Implement when adding authentication
    """
    pass

