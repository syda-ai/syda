# Docker Setup for Syda API

This directory contains Docker-related configuration for the Syda API.

## Files

### `entrypoint.sh`
Entrypoint script that runs when the container starts. It:

1. **Waits for PostgreSQL** to be ready
2. **Runs Alembic migrations** (`alembic upgrade head`)
3. **Starts the FastAPI server** with uvicorn

This ensures the database schema is always up-to-date when the API starts.

## How It Works

The entrypoint script is executed automatically when the container starts:

```dockerfile
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**Flow:**
1. Container starts
2. `entrypoint.sh` executes
3. Waits for PostgreSQL
4. Runs migrations
5. Executes the CMD (starts uvicorn)

## Development

When developing locally, the volume mount allows:
- Code changes trigger hot-reload (via `--reload` flag)
- Database schema changes require rebuilding the container to re-run migrations

## Production

For production deployment:
- Remove `--reload` flag from CMD
- Consider using a production WSGI server (gunicorn + uvicorn workers)
- Set proper environment variables
- Use separate migration step in CI/CD pipeline

## Troubleshooting

### Migrations fail
```bash
# View detailed logs
docker-compose logs api

# Run migrations manually
docker-compose exec api alembic upgrade head

# Check database
docker-compose exec postgres psql -U syda_user -d syda_db
```

### Container keeps restarting
Check if migrations are failing:
```bash
docker-compose logs api | grep -i migration
```

