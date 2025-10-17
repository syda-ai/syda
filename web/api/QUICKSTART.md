# Syda API - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Option 1: Using Docker (Recommended)

```bash
cd web/api

# Start PostgreSQL and API
docker-compose up

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/api/docs
```

### Option 2: Manual Setup

```bash
cd web/api

# 1. Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Setup PostgreSQL
createdb syda_db
createuser syda_user -P  # Set password: syda_password

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Run migrations
alembic upgrade head

# 5. Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Quick Start Script

```bash
cd web/api
./start.sh
```

## 📝 Test the API

### Using the Interactive Docs

Open http://localhost:8000/api/docs in your browser and try:

1. **Create a Provider** (POST `/api/providers`)
```json
{
  "name": "Anthropic",
  "key": "anthropic",
  "api_key": "sk-ant-your-key-here",
  "extra_kwargs": {
    "base_url": "https://api.anthropic.com",
    "timeout": "30"
  }
}
```

2. **List Providers** (GET `/api/providers`)

3. **Test Connection** (POST `/api/providers/{id}/test`)

4. **Create a Workspace** (POST `/api/workspaces`)
```json
{
  "name": "My First Project",
  "description": "Test workspace for synthetic data generation"
}
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/api/health

# Create provider
curl -X POST http://localhost:8000/api/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Anthropic",
    "key": "anthropic",
    "api_key": "sk-ant-test",
    "extra_kwargs": {}
  }'

# List providers
curl http://localhost:8000/api/providers

# Create workspace
curl -X POST http://localhost:8000/api/workspaces \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Project",
    "description": "Test workspace"
  }'
```

## 🔗 Connect Frontend

The React frontend (in `web/ui`) is already configured to connect to this API.

1. **Start the API** (port 8000)
```bash
cd web/api
./start.sh
```

2. **Start the Frontend** (port 5173)
```bash
cd web/ui
npm run dev
```

3. **Open** http://localhost:5173

The frontend will automatically connect to the API at http://localhost:8000

## 📊 Database Schema

The API creates the following tables:

- `workspaces` - Projects/workspaces
- `providers` - AI provider configurations (encrypted API keys)
- `schemas` - Data schema definitions
- `jobs` - Data generation jobs
- `results` - Generated data files
- `settings` - User preferences

## 🔑 API Endpoints

### Providers
- `GET /api/providers` - List all providers
- `POST /api/providers` - Create provider
- `GET /api/providers/{id}` - Get provider
- `PUT /api/providers/{id}` - Update provider
- `DELETE /api/providers/{id}` - Delete provider
- `POST /api/providers/{id}/test` - Test connection

### Workspaces
- `GET /api/workspaces` - List all workspaces
- `POST /api/workspaces` - Create workspace
- `GET /api/workspaces/{id}` - Get workspace
- `PUT /api/workspaces/{id}` - Update workspace
- `DELETE /api/workspaces/{id}` - Delete workspace

### Coming Soon
- Schemas API
- Jobs API
- Results API
- Settings API

## 🛠️ Development

### Create a new migration
```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

### Run tests
```bash
pytest
```

### View logs
```bash
# Docker
docker-compose logs -f api

# Manual
# Check terminal where uvicorn is running
```

## 🐛 Troubleshooting

### Database connection error
- Ensure PostgreSQL is running
- Check DATABASE_URL in .env
- Verify database exists: `psql -U syda_user -d syda_db`

### Import errors
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn main:app --reload --port 8001
```

## 📚 Next Steps

1. ✅ API is running
2. ✅ Database is setup
3. ✅ Providers endpoint working
4. ✅ Workspaces endpoint working
5. 🔜 Implement Schemas API
6. 🔜 Implement Jobs API
7. 🔜 Connect to Syda generation library
8. 🔜 Add authentication

## 🎉 Success!

Your Syda API is now running. Check:
- API: http://localhost:8000
- Docs: http://localhost:8000/api/docs
- Health: http://localhost:8000/api/health

