# Syda - Docker Setup

Complete Docker setup for the entire Syda stack (UI + API + Database).

## 🚀 Quick Start

Start the entire application with one command:

```bash
# From the project root
docker-compose up
```

This will start:
- **PostgreSQL** on `localhost:5432`
- **API (FastAPI)** on `localhost:8000`
- **UI (React/Vite)** on `localhost:5173`

## 🔗 Access Points

- **Frontend (UI)**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **API Health**: http://localhost:8000/api/health
- **PostgreSQL**: `localhost:5432` (syda_db/syda_user/syda_password)

## 📋 Services

### PostgreSQL Database
- **Port**: 5432
- **Database**: syda_db
- **User**: syda_user
- **Password**: syda_password
- **Volume**: Persistent data in `postgres_data` volume

### API (FastAPI Backend)
- **Port**: 8000
- **Hot Reload**: Enabled (code changes auto-reload)
- **Volume**: `./web/api` mounted to `/app`
- **Dependencies**: Automatically installs from `requirements.txt`

### UI (React/Vite Frontend)
- **Port**: 5173
- **Hot Reload**: Enabled (code changes auto-reload)
- **Volume**: `./web/ui` mounted to `/app`
- **Dependencies**: Automatically installs from `package.json`

## 🛠️ Common Commands

### Start all services
```bash
docker-compose up
```

### Start in background (detached mode)
```bash
docker-compose up -d
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f ui
docker-compose logs -f postgres
```

### Stop all services
```bash
docker-compose down
```

### Stop and remove volumes (WARNING: deletes database data)
```bash
docker-compose down -v
```

### Rebuild containers (after dependency changes)
```bash
docker-compose up --build
```

### Restart a specific service
```bash
docker-compose restart api
docker-compose restart ui
```

### Execute commands in containers
```bash
# Access API shell
docker-compose exec api bash

# Access UI shell
docker-compose exec ui sh

# Access PostgreSQL
docker-compose exec postgres psql -U syda_user -d syda_db
```

## 🔧 Database Migrations

Run Alembic migrations:

```bash
# Generate migration
docker-compose exec api alembic revision --autogenerate -m "Description"

# Apply migrations
docker-compose exec api alembic upgrade head

# Rollback migration
docker-compose exec api alembic downgrade -1
```

## 🐛 Troubleshooting

### Port already in use
If you see "port already in use" errors:

```bash
# Find and kill process on port 5432 (PostgreSQL)
lsof -ti:5432 | xargs kill -9

# Find and kill process on port 8000 (API)
lsof -ti:8000 | xargs kill -9

# Find and kill process on port 5173 (UI)
lsof -ti:5173 | xargs kill -9
```

Or change ports in `docker-compose.yml`:
```yaml
ports:
  - "5174:5173"  # Use different external port
```

### Database connection errors
```bash
# Check if PostgreSQL is healthy
docker-compose ps

# View PostgreSQL logs
docker-compose logs postgres

# Recreate database
docker-compose down -v
docker-compose up postgres
```

### Module not found errors (API)
```bash
# Rebuild API container
docker-compose up --build api
```

### Dependency errors (UI)
```bash
# Rebuild UI container
docker-compose up --build ui

# Or manually reinstall
docker-compose exec ui npm install
```

### Clear everything and start fresh
```bash
# Stop and remove everything
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Rebuild and start
docker-compose up --build
```

## 📝 Environment Variables

The `docker-compose.yml` includes default development settings. For production:

1. Create `.env` file in project root:
```env
POSTGRES_DB=syda_db
POSTGRES_USER=syda_user
POSTGRES_PASSWORD=your-secure-password

API_SECRET_KEY=your-secret-key
API_ENCRYPTION_KEY=your-encryption-key-32-chars

VITE_API_URL=https://api.yourdomain.com
```

2. Update `docker-compose.yml` to use env_file:
```yaml
services:
  postgres:
    env_file:
      - .env
  api:
    env_file:
      - .env
```

## 🚀 Development Workflow

### Starting work
```bash
# Start all services
docker-compose up

# Open in browser
# - http://localhost:5173 (UI)
# - http://localhost:8000/api/docs (API docs)
```

### Making changes
- **Frontend**: Edit files in `web/ui/src/` - auto-reloads
- **Backend**: Edit files in `web/api/` - auto-reloads
- **Database**: Migrations via `alembic`

### Testing API
```bash
# Using curl
curl http://localhost:8000/api/health

# Using httpie
http GET http://localhost:8000/api/providers

# Using browser
# Open http://localhost:8000/api/docs
```

## 📦 Production Deployment

For production, you'll want to:

1. **Build production images**:
```bash
# Use production Dockerfiles
# Update docker-compose.prod.yml with optimized settings
```

2. **Use production-ready settings**:
- Disable hot reload
- Set `DEBUG=false`
- Use production database credentials
- Enable HTTPS
- Set proper CORS origins

3. **Use orchestration**:
- Kubernetes
- Docker Swarm
- AWS ECS
- Or managed services (Render, Railway, Fly.io)

## 🎯 Next Steps

1. ✅ Start services: `docker-compose up`
2. ✅ Open UI: http://localhost:5173
3. ✅ Check API: http://localhost:8000/api/docs
4. ✅ Create provider in Settings
5. ✅ Create workspace
6. 🔜 Implement remaining features

## 💡 Tips

- **First time setup**: Initial start might take 2-5 minutes while images build
- **Database persistence**: Data survives container restarts via volumes
- **Hot reload**: Code changes reflect immediately (no rebuild needed)
- **Network isolation**: Services communicate via `syda-network`
- **Health checks**: Services wait for dependencies to be healthy before starting

## 🆘 Getting Help

If you encounter issues:
1. Check logs: `docker-compose logs -f`
2. Check service status: `docker-compose ps`
3. Restart services: `docker-compose restart`
4. Fresh start: `docker-compose down -v && docker-compose up --build`

