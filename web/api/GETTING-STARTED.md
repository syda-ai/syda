# Syda API - Getting Started

## ✅ Your Stack is Running!

All services are operational:

- **PostgreSQL**: http://localhost:5432 (healthy)
- **API**: http://localhost:8000 (healthy)
- **UI**: http://localhost:5173 (running)

## 🎯 What Just Happened

### 1. **Initial Setup** (One-time)
```bash
# Generated initial migration
docker-compose run --rm api alembic revision --autogenerate -m "Initial migration"
```

Created migration file: `alembic/versions/bb0212edefca_initial_migration_create_all_tables.py`

This migration creates all database tables:
- ✅ `providers` - AI provider configurations
- ✅ `workspaces` - Projects/workspaces
- ✅ `schemas` - Data schema definitions
- ✅ `jobs` - Generation jobs
- ✅ `results` - Job results
- ✅ `settings` - User preferences

### 2. **Automatic Migration on Startup**
When you run `docker-compose up` or restart containers:

```
Entrypoint script (docker/entrypoint.sh) runs:
  1. Wait for PostgreSQL ✅
  2. Run: alembic upgrade head ✅ (applies new migrations)
  3. Start API server ✅
```

**No manual commands needed** - migrations apply automatically! 🚀

## 🧪 Verified Working Features

### ✅ Providers API
```bash
# List providers
curl http://localhost:8000/api/providers/

# Create provider
curl -X POST http://localhost:8000/api/providers/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Anthropic",
    "key": "anthropic",
    "api_key": "sk-ant-your-key",
    "extra_kwargs": {
      "base_url": "https://api.anthropic.com",
      "timeout": "30"
    }
  }'
```

**Features:**
- ✅ Encrypted API key storage
- ✅ API key masking in responses
- ✅ Extra kwargs for provider-specific config
- ✅ Status tracking
- ✅ Full CRUD operations

### ✅ Workspaces API
```bash
# List workspaces
curl http://localhost:8000/api/workspaces/

# Create workspace
curl -X POST http://localhost:8000/api/workspaces/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Project",
    "description": "Project description"
  }'
```

## 📚 API Documentation

**Interactive Docs**: http://localhost:8000/api/docs

Here you can:
- 📖 See all available endpoints
- 🧪 Test endpoints interactively
- 📝 View request/response schemas
- 🔍 Try it out with real data

## 🔄 Development Workflow

### Daily Development
```bash
# Start everything
docker-compose up

# Code changes auto-reload:
# - Frontend (web/ui/src/*) → Vite hot-reload
# - Backend (web/api/*.py) → Uvicorn hot-reload
```

### When You Change Database Models

1. **Edit** `web/api/models.py`
   ```python
   # Add a new field
   class Provider(Base):
       # ... existing fields
       new_field = Column(String(100))  # New!
   ```

2. **Generate migration**
   ```bash
   docker-compose exec api alembic revision --autogenerate -m "Add new_field to providers"
   ```

3. **Restart API** (migration applies automatically)
   ```bash
   docker-compose restart api
   ```

4. **Commit migration**
   ```bash
   git add web/api/alembic/versions/
   git commit -m "Add new_field to providers table"
   ```

### View Database
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U syda_user -d syda_db

# List tables
\dt

# Query providers
SELECT * FROM providers;

# Exit
\q
```

## 🎨 Frontend Integration

The React UI is already configured to connect to the API.

**Try it:**
1. Open http://localhost:5173
2. Go to Settings → AI Models
3. Select a provider from dropdown
4. The UI will call `GET /api/providers/`
5. You can configure providers (will call `POST /api/providers/`)

## 📊 Current Status

### ✅ Completed
- FastAPI application with CORS
- PostgreSQL database with migrations
- Automatic migration on container startup
- Providers API (full CRUD + encryption)
- Workspaces API (full CRUD)
- Docker Compose orchestration
- Health checks
- Hot reload for development

### 🔜 To Be Implemented
- Schemas API (CRUD for data schemas)
- Jobs API (data generation jobs)
- Results API (download generated data)
- Settings API (user preferences)
- Connect to Syda core library for actual data generation
- Authentication & authorization

## 🐛 Troubleshooting

### API not starting
```bash
# Check logs
docker-compose logs api

# Common issue: migration failed
# Fix and restart
docker-compose restart api
```

### Database connection errors
```bash
# Check PostgreSQL
docker-compose logs postgres

# Verify connection
docker-compose exec postgres pg_isready -U syda_user -d syda_db
```

### Need to reset database
```bash
# Stop and remove volumes (⚠️ deletes all data!)
docker-compose down -v

# Regenerate migration
docker-compose run --rm api alembic revision --autogenerate -m "Initial migration"

# Start fresh
docker-compose up
```

## 🎯 Next Steps

1. ✅ **API is running** - Test at http://localhost:8000/api/docs
2. ✅ **Database is setup** - Tables created automatically
3. ✅ **Migrations work** - Applied automatically on startup
4. 🔜 **Implement Schemas API** - CRUD for data schemas
5. 🔜 **Implement Jobs API** - Trigger data generation
6. 🔜 **Connect to Syda library** - Use actual data generation
7. 🔜 **Build remaining UI features** - Connect to new APIs

## 📖 Documentation

- **API Docs**: http://localhost:8000/api/docs
- **Full README**: [README.md](README.md)
- **Docker Guide**: [../README-DOCKER.md](../README-DOCKER.md)
- **Web Overview**: [../WEB-README.md](../WEB-README.md)
- **Setup Guide**: [../SETUP.md](../SETUP.md)

---

**🎉 Congratulations! Your Syda full-stack application is running!**

Access points:
- 🎨 **UI**: http://localhost:5173
- 🔌 **API**: http://localhost:8000
- 📚 **Docs**: http://localhost:8000/api/docs
- 🗄️ **Database**: localhost:5432

