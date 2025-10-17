"""
CRUD operations for Workspace model
"""
from sqlalchemy.orm import Session
from models import Workspace
from . import schemas
from typing import List, Optional
import uuid


def get_workspace(db: Session, workspace_id: uuid.UUID) -> Optional[Workspace]:
    """Get a single workspace by ID"""
    return db.query(Workspace).filter(Workspace.id == workspace_id).first()


def get_workspaces(
    db: Session, 
    skip: int = 0, 
    limit: int = 100
) -> List[Workspace]:
    """Get all workspaces"""
    return db.query(Workspace).offset(skip).limit(limit).all()


def get_workspaces_count(db: Session) -> int:
    """Get total count of workspaces"""
    return db.query(Workspace).count()


def create_workspace(db: Session, workspace: schemas.WorkspaceCreate) -> Workspace:
    """Create a new workspace"""
    db_workspace = Workspace(
        name=workspace.name,
        description=workspace.description
    )
    db.add(db_workspace)
    db.commit()
    db.refresh(db_workspace)
    return db_workspace


def update_workspace(
    db: Session, 
    workspace_id: uuid.UUID, 
    workspace_update: schemas.WorkspaceUpdate
) -> Optional[Workspace]:
    """Update a workspace"""
    db_workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not db_workspace:
        return None
    
    update_data = workspace_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_workspace, field, value)
    
    db.commit()
    db.refresh(db_workspace)
    return db_workspace


def delete_workspace(db: Session, workspace_id: uuid.UUID) -> bool:
    """Delete a workspace"""
    db_workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not db_workspace:
        return False
    
    db.delete(db_workspace)
    db.commit()
    return True

