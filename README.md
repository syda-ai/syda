# Synthetic Data Generation Library

A Python-based open-source library for generating synthetic data with AI while preserving referential integrity.

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Core API](#core-api)

  * [Structured Data Generation](#structured-data-generation)
  * [SQLAlchemy Model Integration](#sqlalchemy-model-integration)
  * [Handling Foreign Key Relationships](#handling-foreign-key-relationships)
  * [Automatic Management of Multiple Related Models](#automatic-management-of-multiple-related-models)
  * [Complete CRM Example](#complete-crm-example)
  * [Metadata Enhancement Benefits](#metadata-enhancement-benefits)
  * [Model Selection and Configuration](#model-selection-and-configuration)
  * [Output Options](#output-options)
* [Configuration](#configuration)
* [Contributing](#contributing)
* [License](#license)

## Features

- ## **Synthetic Data Generation**

  * AI-driven sample generation via OpenAI and Anthropic (Claude)
  * Support for both sqlalchemy models and simple dictionary schema maps
- **Referential Integrity**

  * Automatic foreign key detection and resolution
  * Multi-model dependency analysis
- **Custom Generators**

  * Register column- or type-specific functions for specialized data
- **Open Core**

  * Core functionality under AGPL-3.0
  * Premium UI and SaaS features under commercial license

## Installation

Install the package using pip:

```bash
pip install syda
```

## Quick Start

```python
from syda.structured import SyntheticDataGenerator

generator = SyntheticDataGenerator()
schema = {
    'patient_id': 'number',
    'diagnosis_code': 'icd10_code',
    'email': 'email',
    'visit_date': 'date',
    'notes': 'text'
}
prompt = (
    "Generate realistic synthetic patient records with ICD-10 diagnosis codes, "
    "emails, visit dates, and clinical notes."
)

# Generate and save to CSV
output = generator.generate_data(
    schema=schema,
    prompt=prompt,
    sample_size=15,
    output_path='synthetic_output.csv'
)
print(f"Data saved to {output}")
```

## Core API

### Structured Data Generation

Use simple schema maps or SQLAlchemy models to generate data:

```python
from syda.structured import SyntheticDataGenerator

generator = SyntheticDataGenerator(model='gpt-4-turbo')
# Simple dict schema
records = generator.generate_data(
    schema={'id': 'number', 'name': 'text'},
    prompt='Generate user records',
    sample_size=10
)
```

### SQLAlchemy Model Integration

Pass declarative models directly—docstrings and column metadata inform the prompt:

```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from syda.structured import SyntheticDataGenerator

Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String, comment="Full name of the user")

generator = SyntheticDataGenerator()
generator.generate_data(schema=User, prompt='Generate users', sample_size=5)
```

### Handling Foreign Key Relationships

* Detects FKs and labels them as `foreign_key`
* Register custom FK generators sampling parent IDs
* Generate parent before child for referential integrity

```python
# After generating departments:
def dept_fk(row, col): return row['id']
generator.register_generator('foreign_key', dept_fk)
```

### Automatic Management of Multiple Related Models

Simplify multi-table workflows with `generate_related_data`:

```python
results = generator.generate_related_data(
    models=[Customer, Contact, Product, Order, OrderItem],
    prompts={...},
    sample_sizes={...},
    custom_generators={...},
    output_dir='output'
)
```

### Complete CRM Example

See `examples/test_auto_related_models.py` for a full CRM workflow—automatic dependency resolution, prompt enrichment, and integrity checks.

### Metadata Enhancement Benefits

1. **Richer Context**: Uses docstrings, comments, constraints.
2. **Simpler Prompts**: Less manual specification needed.
3. **Constraint Awareness**: Respects `nullable`, `unique`, `length`.
4. **Custom Generators**: Column-level overrides.
5. **Auto Docstring Utilization**: Business context embedded in prompts.

### Model Selection and Configuration

Configure provider, model name, temperature, tokens, and proxy:

```python
from syda.schemas import ModelConfig, ProxyConfig
config = ModelConfig(
    provider='openai', model_name='gpt-4', temperature=0.8,
    proxy=ProxyConfig(base_url='https://ai-proxy/', headers={'X-Key':'token'})
)
generator = SyntheticDataGenerator(model_config=config)
```

### Output Options

* Return a `pandas.DataFrame`
* Save to `.csv` or `.json` if `output_path` ends accordingly

## Configuration

Use environment variables or pass API keys directly:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Or in code:

```python
generator = SyntheticDataGenerator(
    openai_api_key='...', anthropic_api_key='...'
)
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Push branch
5. Open a PR

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details.
