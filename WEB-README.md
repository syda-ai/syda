# Syda Web Application

Full-stack web application for Syda synthetic data generation platform.

## 🏗️ Architecture

```
syda-fresh/
├── web/
│   ├── ui/          # React + TypeScript + Vite frontend
│   └── api/         # FastAPI + PostgreSQL backend
├── syda/            # Core Python library
└── docker-compose.yml  # Full stack orchestration
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

Start the entire stack with one command:

```bash
# From project root
docker-compose up

# Or use the helper script
./start-dev.sh
```

**Access points:**
- **UI**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs

### Option 2: Manual Development

#### 1. Start PostgreSQL
```bash
# Using Docker
docker run -d \
  --name syda_postgres \
  -e POSTGRES_DB=syda_db \
  -e POSTGRES_USER=syda_user \
  -e POSTGRES_PASSWORD=syda_password \
  -p 5432:5432 \
  postgres:15-alpine

# Or install PostgreSQL locally
```

#### 2. Start Backend (API)
```bash
cd web/api

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Run migrations
alembic upgrade head

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Start Frontend (UI)
```bash
cd web/ui

# Setup
npm install

# Start dev server
npm run dev
```

## 📦 Technology Stack

### Frontend (`web/ui`)
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **TanStack Router** - Routing
- **TanStack Query** - Data fetching
- **CSS Variables** - Theming (dark/light)

### Backend (`web/api`)
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Relational database
- **SQLAlchemy** - ORM
- **Alembic** - Database migrations
- **Pydantic** - Data validation
- **Cryptography** - API key encryption

## 🎨 Features

### Completed ✅
- **Settings Page** - AI provider management with encrypted API key storage
- **Provider API** - Full CRUD for AI providers (Anthropic, OpenAI, Gemini, Azure OpenAI, Grok)
- **Workspace API** - Project organization
- **Dark/Light Theme** - User preference with persistent storage
- **Dependency Graph** - Visualize schema relationships (UI ready)
- **Responsive Design** - Works on desktop, tablet, mobile

### In Progress 🔨
- Schema management API
- Data generation jobs API
- Results download API
- Settings persistence API

### Planned 🔜
- Authentication & authorization
- Real-time job progress
- Schema validation
- Advanced dependency analysis
- Batch operations
- Export/import configurations

## 📚 Project Structure

### Frontend Structure
```
web/ui/src/
├── features/           # Feature modules
│   ├── settings/       # Settings page (✅ Complete)
│   ├── deps/           # Dependency graph
│   ├── schemas/        # Schema management
│   └── jobs/           # Job management
├── store/              # Global state
│   └── ThemeContext    # Theme management
├── components/         # Reusable components
└── styles/             # Global styles
```

### Backend Structure
```
web/api/
├── main.py            # FastAPI app
├── models.py          # SQLAlchemy models
├── database.py        # DB setup
├── providers/         # Providers feature (✅ Complete)
│   ├── router.py      # API endpoints
│   ├── schemas.py     # Pydantic models
│   ├── crud.py        # DB operations
│   └── service.py     # Business logic
├── workspaces/        # Workspaces feature (✅ Complete)
└── utils/             # Security, validation
```

## 🔌 API Endpoints

### Providers
- `GET /api/providers` - List all providers
- `GET /api/providers/{id}` - Get provider details
- `POST /api/providers` - Create provider
- `PUT /api/providers/{id}` - Update provider
- `DELETE /api/providers/{id}` - Delete provider
- `POST /api/providers/{id}/test` - Test connection

### Workspaces
- `GET /api/workspaces` - List workspaces
- `GET /api/workspaces/{id}` - Get workspace
- `POST /api/workspaces` - Create workspace
- `PUT /api/workspaces/{id}` - Update workspace
- `DELETE /api/workspaces/{id}` - Delete workspace

### Coming Soon
- `/api/schemas` - Schema management
- `/api/jobs` - Generation jobs
- `/api/results` - Download results
- `/api/settings` - User settings

## 🔐 Security

- **API Key Encryption**: Provider API keys encrypted using Fernet (symmetric encryption)
- **API Key Masking**: Keys masked in API responses
- **CORS**: Configured for specific origins
- **Input Validation**: Pydantic schemas validate all inputs
- **SQL Injection Protection**: SQLAlchemy ORM prevents SQL injection
- **Environment Variables**: Sensitive config in .env files

## 🧪 Testing

### Frontend
```bash
cd web/ui
npm test              # Run tests
npm run test:coverage # With coverage
```

### Backend
```bash
cd web/api
pytest                # Run tests
pytest --cov          # With coverage
```

## 🛠️ Development

### Adding a New Feature

**Backend:**
1. Create feature folder: `web/api/feature_name/`
2. Add: `router.py`, `schemas.py`, `crud.py`, `service.py`
3. Register router in `main.py`
4. Create migration: `alembic revision --autogenerate -m "Add feature"`
5. Apply: `alembic upgrade head`

**Frontend:**
1. Create feature folder: `web/ui/src/features/feature_name/`
2. Add components, hooks, and API client
3. Add route if needed
4. Update navigation

### Database Migrations

```bash
cd web/api

# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1

# View history
alembic history
```

## 📊 Database Schema

**Tables:**
- `workspaces` - Projects/workspaces
- `providers` - AI provider configurations
- `schemas` - Data schema definitions
- `jobs` - Generation jobs
- `results` - Generated data
- `settings` - User preferences

See `web/api/models.py` for full schema.

## 🐛 Troubleshooting

### Frontend issues
```bash
# Clear cache
rm -rf web/ui/node_modules web/ui/dist
npm install

# Check Vite config
cat web/ui/vite.config.ts
```

### Backend issues
```bash
# Reinstall dependencies
cd web/api
pip install -r requirements.txt

# Check database connection
python -c "from database import engine; print(engine.url)"

# Reset database
alembic downgrade base
alembic upgrade head
```

### Docker issues
```bash
# Clean rebuild
docker-compose down -v
docker-compose up --build

# View logs
docker-compose logs -f api
docker-compose logs -f ui
```

## 📖 Documentation

- **Frontend**: See `web/ui/README.md`
- **Backend**: See `web/api/README.md`
- **Docker**: See `README-DOCKER.md`
- **API Docs**: http://localhost:8000/api/docs (when running)

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Add tests
4. Update documentation
5. Submit PR

## 📝 License

MIT License - See LICENSE file

## 🎯 Roadmap

- [x] Backend API structure
- [x] Frontend UI framework
- [x] Provider management
- [x] Workspace management
- [x] Theme support
- [ ] Schema management
- [ ] Job execution
- [ ] Authentication
- [ ] WebSocket for real-time updates
- [ ] File upload/download
- [ ] Advanced analytics

---

**Need help?** Check the docs or open an issue!

