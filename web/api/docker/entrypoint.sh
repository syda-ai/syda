#!/bin/bash
set -e

echo "🔄 Syda API - Starting initialization..."

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until pg_isready -h postgres -p 5432 -U syda_user; do
  echo "   Postgres is unavailable - sleeping"
  sleep 1
done
echo "✅ PostgreSQL is ready"

# Run database migrations
echo "🔄 Running Alembic migrations..."
alembic upgrade head

if [ $? -eq 0 ]; then
  echo "✅ Migrations applied successfully"
else
  echo "❌ Migrations failed"
  exit 1
fi

# Start the application (exec replaces the shell process with the command)
echo "🚀 Starting Syda API server..."
echo "   Listening on http://0.0.0.0:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""

exec "$@"

