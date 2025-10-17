"""
Configuration settings for the API
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "postgresql://syda_user:syda_password@localhost:5432/syda_db"
    
    # Security
    secret_key: str = "change-this-to-a-secure-random-key-in-production"
    encryption_key: str = "change-this-to-32-byte-key"  # For API key encryption
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # API
    api_title: str = "Syda API"
    api_version: str = "1.0.0"
    
    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

