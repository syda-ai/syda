# Syda API

FastAPI backend for Syda synthetic data generation platform.

## Features

- 🔐 **Secure Provider Management** - Encrypted storage of AI provider API keys
- 📊 **Schema Management** - Define and manage data schemas
- ⚙️ **Job Management** - Queue and track data generation jobs
- 🗂️ **Workspace Organization** - Organize schemas and jobs by project
- 🔄 **Dependency Tracking** - Visualize schema relationships

## Tech Stack

- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Relational database
- **SQLAlchemy** - ORM
- **Alembic** - Database migrations
- **Pydantic** - Data validation
- **Cryptography** - API key encryption

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip or poetry

### Installation

1. **Clone the repository** (if not already done)

2. **Navigate to API directory**
   ```bash
   cd web/api
   ```

3. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Setup PostgreSQL database**
   ```bash
   # Create database
   createdb syda_db
   
   # Or using psql
   psql -U postgres
   CREATE DATABASE syda_db;
   CREATE USER syda_user WITH PASSWORD 'syda_password';
   GRANT ALL PRIVILEGES ON DATABASE syda_db TO syda_user;
   ```

7. **Initialize database**
   ```bash
   # Option 1: Using Alembic (recommended)
   alembic upgrade head
   
   # Option 2: Direct creation (for development)
   python -c "from database import init_db; init_db()"
   ```

8. **Run the server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

9. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/api/docs
   - ReDoc: http://localhost:8000/api/redoc

## API Endpoints

### Providers
- `GET /api/providers` - List all providers
- `GET /api/providers/{id}` - Get provider details
- `POST /api/providers` - Create provider
- `PUT /api/providers/{id}` - Update provider
- `DELETE /api/providers/{id}` - Delete provider
- `POST /api/providers/{id}/test` - Test connection

### Workspaces (Coming soon)
- `GET /api/workspaces` - List workspaces
- `POST /api/workspaces` - Create workspace
- etc.

### Schemas (Coming soon)
- `GET /api/schemas` - List schemas
- `POST /api/schemas` - Create schema
- etc.

### Jobs (Coming soon)
- `GET /api/jobs` - List jobs
- `POST /api/jobs` - Create job
- etc.

## Project Structure

```
web/api/
├── main.py                 # FastAPI app
├── config.py               # Configuration
├── database.py             # Database setup
├── models.py               # SQLAlchemy models
├── dependencies.py         # Common dependencies
│
├── providers/              # Provider management
│   ├── router.py           # API routes
│   ├── schemas.py          # Pydantic models
│   ├── crud.py             # Database operations
│   └── service.py          # Business logic
│
├── utils/                  # Utilities
│   ├── security.py         # Encryption
│   └── validators.py       # Validators
│
├── alembic/                # Database migrations
│   └── versions/
│
├── requirements.txt
├── .env.example
└── README.md
```

## Development

### Running tests
```bash
pytest
```

### Create a new migration
```bash
alembic revision --autogenerate -m "Description"
```

### Apply migrations
```bash
alembic upgrade head
```

### Rollback migration
```bash
alembic downgrade -1
```

## Environment Variables

See `.env.example` for all available configuration options.

Key variables:
- `DATABASE_URL` - PostgreSQL connection string
- `SECRET_KEY` - Secret key for security
- `ENCRYPTION_KEY` - Key for encrypting API keys
- `ENVIRONMENT` - development/production
- `DEBUG` - Enable debug mode

## Security

- API keys are encrypted using Fernet (symmetric encryption)
- All sensitive data is masked in API responses
- CORS is configured for specified origins only
- TODO: Add authentication/authorization

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## License

MIT License - See parent project for details

