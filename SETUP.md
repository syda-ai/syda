# Syda Full Stack Setup Guide

Complete setup guide for Syda web application (UI + API + Database).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
├──────────────┬──────────────────┬──────────────────────┤
│  PostgreSQL  │   FastAPI API    │   React/Vite UI      │
│   :5432      │     :8000        │      :5173           │
└──────────────┴──────────────────┴──────────────────────┘
```

## 🚀 Quick Start (Recommended)

### Prerequisites
- Docker & Docker Compose installed
- Ports 5432, 8000, 5173 available

### Start Everything

```bash
# From project root
docker-compose up

# Or use the start script
./start-dev.sh
```

**That's it!** The setup automatically:
1. ✅ Starts PostgreSQL
2. ✅ Waits for database to be ready
3. ✅ **Runs Alembic migrations automatically** 🎯
4. ✅ Starts FastAPI server
5. ✅ Starts React UI

### Access Points
- **UI**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **Database**: localhost:5432 (user: syda_user, db: syda_db)

## 🔄 How Migrations Work (Like LiteLLM)

Following [LiteLLM's pattern](https://github.com/BerriAI/litellm), migrations run **automatically via entrypoint script**:

### Automatic Migration Flow

```
docker-compose up
    ↓
PostgreSQL starts → Health check passes
    ↓
API container starts
    ↓
Entrypoint script runs (web/api/docker/entrypoint.sh)
    ↓
1. Wait for PostgreSQL (pg_isready check)
    ↓
2. Run migrations (alembic upgrade head)  ← Automatic!
    ↓
3. Start API server (uvicorn main:app)
    ↓
✅ API ready at http://localhost:8000
```

### Entrypoint Script (`web/api/docker/entrypoint.sh`)

```bash
#!/bin/bash
set -e

echo "🔄 Syda API - Starting initialization..."

# Wait for PostgreSQL
until pg_isready -h postgres -p 5432 -U syda_user; do
  echo "   Postgres is unavailable - sleeping"
  sleep 1
done
echo "✅ PostgreSQL is ready"

# Run migrations (AUTOMATIC!)
echo "🔄 Running Alembic migrations..."
alembic upgrade head
echo "✅ Migrations applied successfully"

# Start server
echo "🚀 Starting Syda API server..."
exec "$@"
```

### Why This Approach?

✅ **Automatic** - No manual migration commands needed
✅ **Idempotent** - Safe to run multiple times (Alembic only applies new migrations)
✅ **Production-ready** - Used by LiteLLM and other major projects
✅ **Developer-friendly** - Just `docker-compose up` and it works
✅ **Fail-safe** - Container exits if migrations fail

### Creating New Migrations

When you change database models:

```bash
# Generate migration
docker-compose exec api alembic revision --autogenerate -m "Add new table"

# Restart API to apply (or wait for hot-reload)
docker-compose restart api

# Migration runs automatically on startup!
```

## 📁 Project Structure

```
syda-fresh/
├── docker-compose.yml          # Full stack orchestration
├── start-dev.sh                # Quick start script
├── test-api.sh                 # API test script
│
├── web/
│   ├── ui/                     # React Frontend
│   │   ├── Dockerfile
│   │   ├── src/
│   │   └── package.json
│   │
│   └── api/                    # FastAPI Backend
│       ├── Dockerfile
│       ├── docker/
│       │   └── entrypoint.sh   # 🎯 Runs migrations!
│       ├── main.py
│       ├── models.py
│       ├── database.py
│       ├── providers/          # Provider feature
│       ├── workspaces/         # Workspace feature
│       ├── alembic/            # Migrations
│       └── requirements.txt
│
└── syda/                       # Core Python library
```

## 🛠️ Development Workflow

### Daily Development

```bash
# Start everything
docker-compose up

# Make code changes - auto-reload happens!
# Edit web/ui/src/* - Vite hot-reloads
# Edit web/api/*.py - Uvicorn hot-reloads
```

### Database Changes

```bash
# 1. Edit models.py
vim web/api/models.py

# 2. Generate migration
docker-compose exec api alembic revision --autogenerate -m "Your change"

# 3. Restart API (migration runs automatically)
docker-compose restart api

# OR just wait - migrations run on next container start
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f ui
docker-compose logs -f postgres
```

### Access Database

```bash
# Using psql
docker-compose exec postgres psql -U syda_user -d syda_db

# List tables
docker-compose exec postgres psql -U syda_user -d syda_db -c '\dt'

# Query providers
docker-compose exec postgres psql -U syda_user -d syda_db -c 'SELECT * FROM providers;'
```

## 🧪 Testing

### Test API Endpoints

```bash
# Run test script
./test-api.sh

# Or manually
curl http://localhost:8000/api/health
curl http://localhost:8000/api/providers
curl http://localhost:8000/api/workspaces
```

### Test Provider Creation

```bash
curl -X POST http://localhost:8000/api/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Anthropic",
    "key": "anthropic",
    "api_key": "sk-ant-test-key",
    "extra_kwargs": {
      "base_url": "https://api.anthropic.com",
      "timeout": "30"
    }
  }'
```

## 🔧 Environment Variables

Set these in `.env` file or export them:

```bash
# Database
DATABASE_URL=postgresql://syda_user:syda_password@postgres:5432/syda_db

# Security (change in production!)
SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-32-char-encryption-key

# API
ENVIRONMENT=development
DEBUG=true
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# UI
VITE_API_URL=http://localhost:8000
```

## 🚢 Production Deployment

### 1. Build Production Images

```bash
# Build optimized images
docker-compose -f docker-compose.prod.yml build
```

### 2. Update Environment

Create `.env.production`:
```env
DATABASE_URL=postgresql://user:pass@prod-host:5432/syda_db
SECRET_KEY=<strong-random-key>
ENCRYPTION_KEY=<strong-random-32-char-key>
ENVIRONMENT=production
DEBUG=false
```

### 3. Deploy

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🐛 Troubleshooting

### Migrations Don't Run

Check entrypoint logs:
```bash
docker-compose logs api | grep -i migration
```

Manually run migrations:
```bash
docker-compose exec api alembic upgrade head
```

### Port Conflicts

Change ports in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 externally instead of 8000
```

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U syda_user -d syda_db
```

### Reset Everything

```bash
# Stop and remove all containers + volumes
docker-compose down -v

# Rebuild and start fresh
docker-compose up --build
```

## 📊 Monitoring

### Container Status

```bash
docker-compose ps
```

### Resource Usage

```bash
docker stats
```

### Database Size

```bash
docker-compose exec postgres psql -U syda_user -d syda_db -c "
  SELECT pg_size_pretty(pg_database_size('syda_db'));
"
```

## 🎯 Next Steps

1. ✅ Docker setup complete
2. ✅ Migrations run automatically
3. ✅ Providers & Workspaces API working
4. 🔜 Implement Schemas API
5. 🔜 Implement Jobs API
6. 🔜 Connect to Syda core library
7. 🔜 Add authentication

---

**Questions?** Check:
- [Web README](WEB-README.md) - Web application overview
- [Docker README](README-DOCKER.md) - Docker details
- [API README](web/api/README.md) - Backend details
- [UI README](web/ui/README.md) - Frontend details

