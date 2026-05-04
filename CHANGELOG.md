# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.5] - 2026-05-03

### Added
- `DatabaseSchemaLoader` class in `syda/db_schema_loader.py` for connecting directly to relational databases
  - `load_schemas()` — infers table schemas (columns, types, primary keys, foreign keys) as Python dicts
  - `save_schemas()` — writes one YAML or JSON schema file per table to disk
  - `write_to_database()` — inserts generated DataFrames back into the database in FK-safe topological order
- Database integration examples in `examples/database_integration/`:
  - `example_load_schemas.py` — in-memory schema workflow (SQLite)
  - `example_save_schemas.py` — file-based schema workflow (SQLite)
  - `example_postgres.py` — full PostgreSQL end-to-end example
  - `healthcare_demo.db` — SQLite demo database with patient, provider, diagnosis, claim, payment, and adjudication tables
  - Pre-generated schema YAML files and sample output CSVs
- `tests/test_db_schema.py` — comprehensive test suite for `DatabaseSchemaLoader`
- `docs/deep_dive/database_integration.md` — full documentation covering supported databases, Option A/B workflows, type mapping, FK-safe insertion order, and API reference
- `scripts/pre_release_test.sh` — automated pre-release validation script
- `CLAUDE.md` — developer guidance file for Claude Code
- PostgreSQL (`psycopg2-binary`) and MySQL (`pymysql`) driver dependencies in `requirements.txt`
- `DatabaseSchemaLoader` exported from `syda/__init__.py`

### Changed
- Version bumped to `0.0.5`
- `README.md` updated with database integration overview and usage examples
- Database Integration page added to MkDocs navigation (`deep_dive/database_integration`)

## [0.0.3] - 2025-09-21

### Added
- Azure OpenAI provider support for enterprise deployments
- Advanced configuration with `extra_kwargs` parameter for all providers
- AI gateway integration support (LiteLLM, Portkey, Kong, and custom gateways)
- Comprehensive Azure OpenAI documentation and examples
- Enhanced model configuration guide with `extra_kwargs` reference
- Support for custom endpoints, authentication headers, and timeouts
- Enterprise-grade features for production deployments

### Changed
- Development status upgraded from Beta to Production/Stable
- Enhanced documentation with AI gateway integration examples
- Improved error handling and troubleshooting guidance
- Updated model configuration documentation with provider-specific examples

### Fixed
- Enhanced provider-specific parameter handling
- Better error messages for configuration issues


## [0.0.2] - 2025-08-23

### Added
- Support for Google Gemini Models

### Changed
- Documentation Fixes


## [0.0.1] - 2025-08-11

### Added
- Modern packaging with pyproject.toml
- Support for multiple AI providers (OpenAI, Anthropic Claude)
- Comprehensive schema formats (SQLAlchemy, YAML, JSON, Dict)
- Foreign key relationship handling with referential integrity
- Unstructured document generation with templates
- Custom generators for domain-specific data
- Multi-provider AI integration with consistent interface
- Automatic dependency resolution via topological sorting