"""
Syda API - FastAPI application for synthetic data generation
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers (will be uncommented as we create them)
from providers.router import router as providers_router
from workspaces.router import router as workspaces_router
# from schemas.router import router as schemas_router
# from jobs.router import router as jobs_router
# from results.router import router as results_router
# from settings.router import router as settings_router

app = FastAPI(
    title="Syda API",
    description="API for Syda synthetic data generation with AI providers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Import config
from config import settings

# CORS middleware - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(providers_router, prefix="/providers", tags=["providers"])
app.include_router(workspaces_router, prefix="/workspaces", tags=["workspaces"])
# app.include_router(schemas_router, prefix="/schemas", tags=["schemas"])
# app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
# app.include_router(results_router, prefix="/results", tags=["results"])
# app.include_router(settings_router, prefix="/settings", tags=["settings"])


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Syda API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

