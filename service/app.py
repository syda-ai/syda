from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import pandas as pd
from io import StringIO, BytesIO
import tempfile
from pathlib import Path
from syda.structured import SyntheticDataGenerator
from service.db.database import SessionLocal, engine
from service.db.models import Base, Project, Transaction
import json
from datetime import datetime
import os
import openai
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize AI clients
openai.api_key = os.getenv("OPENAI_API_KEY")
claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Powered Synthetic Data Generation Service",
    description="Generate realistic synthetic data using advanced AI models",
    version="1.0.0"
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ProjectCreate(BaseModel):
    name: str = Field(..., description="Name of the project")
    description: Optional[str] = Field(None, description="Optional project description")

class Project(ProjectCreate):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True



class AIGenerationRequest(BaseModel):
    project_name: str = Field(..., description="Name of the project for tracking")
    prompt: str = Field(..., description="Description of the data to generate")
    schema: Dict[str, str] = Field(..., description="Schema of the data to generate")
    n_samples: int = Field(100, description="Number of samples to generate", ge=1, le=1000)
    model: str = Field("gpt-4", description="AI model to use (gpt-4, claude-2, etc.)")
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=1.0)

class FileGenerationRequest(BaseModel):
    project_name: str = Field(..., description="Name of the project for tracking")
    file: UploadFile = File(..., description="Upload a CSV or Excel file")
    n_samples: int = Field(100, description="Number of samples to generate", ge=1, le=1000)
    model: str = Field("gpt-4", description="AI model to use (gpt-4, claude-2, etc.)")
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=1.0)

def create_transaction(db, project, operation_type, input_data):
    """Helper function to create a transaction record"""
    transaction = Transaction(
        project_id=project.id,
        operation_type=operation_type,
        input_data=json.dumps(input_data),
        output_data='pending',
        status='processing'
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction



@app.post("/generate/ai", response_model=List[Dict])
async def generate_ai_data(request: AIGenerationRequest):
    """
    Generate synthetic data using AI models
    
    Request:
    - project_name: Name of the project
    - prompt: Description of the data to generate
    - schema: Dictionary mapping field names to their types
    - n_samples: Number of samples to generate (1-1000)
    - model: AI model to use (gpt-4, claude-2, etc.)
    - temperature: Sampling temperature (0.0-1.0)
    
    Returns:
    - List of generated data samples
    """
    db = next(get_db())
    
    try:
        # Get or create project
        project = db.query(Project).filter(Project.name == request.project_name).first()
        if not project:
            project = Project(name=request.project_name, description="Auto-created")
            db.add(project)
            db.commit()
            db.refresh(project)
        
        # Create transaction record
        transaction = create_transaction(db, project, 'ai_generate', {
            "prompt": request.prompt,
            "schema": request.schema,
            "n_samples": request.n_samples,
            "model": request.model,
            "temperature": request.temperature
        })
        
        # Generate data using AI
        generated_data = generate_with_ai(
            schema=request.schema,
            prompt=request.prompt,
            n_samples=request.n_samples,
            model=request.model,
            temperature=request.temperature
        )
        
        # Update transaction
        transaction.status = 'completed'
        transaction.output_data = json.dumps(generated_data)
        db.commit()
        
        return generated_data
        
    except Exception as e:
        if 'transaction' in locals() and transaction:
            transaction.status = 'failed'
            transaction.error_message = str(e)
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/test-data")
async def generate_test_data(request: AIGenerationRequest):
    """
    Generate test data based on schema using AI
    
    Request Body:
    {
        "project_name": "project_name",
        "prompt": "Description of the data to generate",
        "schema": {
            "field1": "type1",
            "field2": "type2"
        },
        "n_samples": 50,
        "model": "gpt-4",
        "temperature": 0.7
    }
    """
    return await generate_ai_data(request)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI-Powered Synthetic Data Generation Service",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/generate/ai", "method": "POST", "description": "Generate synthetic data using AI models"},
            {"path": "/generate/file", "method": "POST", "description": "Generate synthetic data from uploaded file"},
            {"path": "/generate/test-data", "method": "POST", "description": "Generate test data based on schema"}
        ]
    }

@app.post("/generate/file", response_model=List[Dict])
async def generate_from_file(
    project_name: str = Query(..., description="Name of the project"),
    file: UploadFile = File(..., description="Upload a CSV or Excel file"),
    n_samples: int = Query(100, description="Number of samples to generate", ge=1, le=1000),
    model: str = Query("gpt-4", description="AI model to use (gpt-4, claude-2, etc.)"),
    temperature: float = Query(0.7, description="Sampling temperature", ge=0.0, le=1.0)
):
    """
    Generate synthetic data from an uploaded file
    
    Request:
    - project_name: Name of the project
    - file: Uploaded file (CSV or Excel)
    - n_samples: Number of samples to generate (1-1000)
    - model: AI model to use (gpt-4, claude-2, etc.)
    - temperature: Sampling temperature (0.0-1.0)
    
    Returns:
    - List of generated data samples
    """
    db = next(get_db())
    
    try:
        # Get or create project
        project = db.query(Project).filter(Project.name == project_name).first()
        if not project:
            project = Project(name=project_name, description="Auto-created")
            db.add(project)
            db.commit()
            db.refresh(project)
        
        # Read file
        if file.filename.endswith('.csv'):
            contents = await file.read()
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            contents = await file.read()
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create schema from file
        schema = {col: "string" for col in df.columns}
        
        # Create transaction record
        transaction = create_transaction(
            db=db,
            project=project,
            operation_type='file_generate',
            input_data={
                "filename": file.filename,
                "columns": list(df.columns),
                "n_samples": n_samples,
                "model": model,
                "temperature": temperature
            }
        )
        
        try:
            # Generate data using AI
            prompt = f"Generate {n_samples} synthetic samples similar to the data in the uploaded file."
            generated_data = generate_with_ai(
                schema=schema,
                prompt=prompt,
                n_samples=n_samples,
                model=model,
                temperature=temperature
            )
            
            # Update transaction
            transaction.status = 'completed'
            transaction.output_data = json.dumps(generated_data)
            db.commit()
            
            return generated_data
            
        except Exception as e:
            transaction.status = 'failed'
            transaction.error_message = str(e)
            db.commit()
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects")
async def create_project(project: ProjectCreate, db: SessionLocal = Depends(get_db)):
    """
    Create a new project
    
    Request Body:
    {
        "name": "project_name",
        "description": "optional description"
    }
    """
    # Check if project with same name exists
    existing_project = db.query(Project).filter(Project.name == project.name).first()
    if existing_project:
        raise HTTPException(status_code=400, detail="Project with this name already exists")
    
    db_project = Project(
        name=project.name,
        description=project.description
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    return db_project

@app.get("/projects")
async def get_projects(db: SessionLocal = Depends(get_db)):
    """
    Get all projects
    """
    projects = db.query(Project).all()
    return projects

@app.get("/projects/{project_name}/transactions")
async def get_project_transactions(project_name: str, db: SessionLocal = Depends(get_db)):
    """
    Get transactions for a specific project
    """
    project = db.query(Project).filter(Project.name == project_name).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    transactions = db.query(Transaction).filter(Transaction.project_id == project.id).all()
    return [
        {
            "id": t.id,
            "operation_type": t.operation_type,
            "created_at": t.created_at.isoformat(),
            "status": t.status,
            "input_data": t.input_data,
            "output_data": t.output_data
        }
        for t in transactions
    ]

@app.post("/upload/unstructured")
async def process_unstructured_file(file: UploadFile = File(...), project_name: str = None):
    """
    Process unstructured file (image, PDF, Word, Excel, text)
    
    Request:
    - Upload a file (supported types: image, PDF, Word, Excel, text)
    - Optional: project_name parameter
    """
    try:
        db = next(get_db())
        
        # Get project if provided
        project = None
        if project_name:
            project = db.query(Project).filter(Project.name == project_name).first()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the file
        processor = UnstructuredDataProcessor()
        result = processor.process_file(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Create transaction record if project is provided
        if project:
            transaction = Transaction(
                project_id=project.id,
                operation_type='process_unstructured',
                input_data=json.dumps({
                    "file_type": result.get('type'),
                    "file_name": file.filename
                }),
                output_data=json.dumps(result),
                status='completed'
            )
            db.add(transaction)
            db.commit()
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/unstructured/generate")
async def generate_synthetic_from_unstructured_file(file: UploadFile = File(...), n_samples: int = 100, project_name: str = None):
    """
    Generate synthetic data from unstructured file
    
    Request:
    - Upload a file (supported types: Excel)
    - Optional: n_samples parameter
    - Optional: project_name parameter
    """
    try:
        db = next(get_db())
        
        # Get project if provided
        project = None
        if project_name:
            project = db.query(Project).filter(Project.name == project_name).first()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the file
        processor = UnstructuredDataProcessor()
        result = processor.generate_synthetic_file(temp_file_path, n_samples)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Create transaction record if project is provided
        if project:
            transaction = Transaction(
                project_id=project.id,
                operation_type='generate_unstructured',
                input_data=json.dumps({
                    "file_type": result.get('type'),
                    "file_name": file.filename,
                    "n_samples": n_samples
                }),
                output_data=json.dumps(result),
                status='completed'
            )
            db.add(transaction)
            db.commit()
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/generate")
async def generate_from_uploaded_file(
    file: UploadFile = File(...),
    n_samples: int = 100,
    project_name: str = None
):
    """
    Generate synthetic data from uploaded CSV file
    
    Request:
    - Upload a CSV file
    - Optional: n_samples parameter
    """
    try:
        db = next(get_db())
        
        # Get or create project if project_name is provided
        project = None
        if project_name:
            project = db.query(Project).filter(Project.name == project_name).first()
            if not project:
                project = Project(name=project_name, description="Auto-created from file upload")
                db.add(project)
                db.commit()
                db.refresh(project)
        
        # Create transaction record if project exists
        transaction = None
        if project:
            transaction = Transaction(
                project_id=project.id,
                operation_type='generate_file',
                input_data=file.filename,
                output_data='pending',
                status='processing'
            )
            db.add(transaction)
            db.commit()
            db.refresh(transaction)
        
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Generate synthetic data (simple example - just sample from existing data)
        if len(df) > 0:
            synthetic_df = df.sample(n=min(n_samples, len(df)*2), replace=True, ignore_index=True)
            
            # Update transaction if exists
            if transaction:
                transaction.status = 'completed'
                transaction.output_data = f'Generated {len(synthetic_df)} synthetic samples'
                db.commit()
            
            # Convert to CSV and return
            csv_data = StringIO()
            synthetic_df.to_csv(csv_data, index=False)
            return Response(
                content=csv_data.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=synthetic_{file.filename}"}
            )
        else:
            if transaction:
                transaction.status = 'failed'
                transaction.error_message = 'No data in input file'
                db.commit()
            raise HTTPException(status_code=400, detail="No data in input file")
    except Exception as e:
        if 'transaction' in locals() and transaction:
            transaction.status = 'failed'
            transaction.error_message = str(e)
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))
