"""
API routes for Workspace management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from database import get_db
from . import schemas, crud
import uuid

router = APIRouter()


@router.get("/", response_model=schemas.WorkspaceListResponse)
def list_workspaces(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all workspaces"""
    workspaces = crud.get_workspaces(db, skip=skip, limit=limit)
    total = crud.get_workspaces_count(db)
    return schemas.WorkspaceListResponse(workspaces=workspaces, total=total)


@router.get("/{workspace_id}", response_model=schemas.WorkspaceResponse)
def get_workspace(workspace_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get a specific workspace by ID"""
    workspace = crud.get_workspace(db, workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@router.post("/", response_model=schemas.WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(workspace: schemas.WorkspaceCreate, db: Session = Depends(get_db)):
    """Create a new workspace"""
    return crud.create_workspace(db, workspace)


@router.put("/{workspace_id}", response_model=schemas.WorkspaceResponse)
def update_workspace(
    workspace_id: uuid.UUID,
    workspace: schemas.WorkspaceUpdate,
    db: Session = Depends(get_db)
):
    """Update a workspace"""
    updated = crud.update_workspace(db, workspace_id, workspace)
    if not updated:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return updated


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(workspace_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete a workspace"""
    success = crud.delete_workspace(db, workspace_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return None

