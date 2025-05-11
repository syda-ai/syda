
# Data Masking and Synthetic Data Generation Service -- WORK IN PROGRESS

A Python-based open-source service for data masking and generating synthetic data while preserving data utility.

## Features

### Core Features

- **Synthetic Data Generation**:
  - Statistical data generation
  - Pattern-based generation
  - Data distribution preservation
  - Synthetic data from unstructured files

### Optional Features

- **REST API Service**:
  - Mask data via API
  - Generate synthetic data via API
  - Support for CSV file uploads
  - Support for unstructured files (images, PDFs, documents)
  - JSON and CSV output formats

- **Project Management**:
  - Create and manage projects
  - Unique project names
  - Project-based transaction tracking
  - Project descriptions

- **Database Integration**:
  - PostgreSQL backend
  - Transaction logging
  - Audit trail for all operations

## Core Module Usage

### Installation

Install the package using pip:

```bash
pip install syda
```

### Basic Usage

#### Synthetic Data Generation

##### Basic Usage

You can generate synthetic data and write it directly to a CSV file using the `output_path` argument:

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
prompt = "Generate realistic synthetic patient records with ICD-10 diagnosis codes, emails, visit dates, and clinical notes."

output_path = 'synthetic_output.csv'
generated_file = generator.generate_data(
    schema=schema,
    prompt=prompt,
    sample_size=15,
    output_path=output_path
)
print(f"Synthetic data written to: {generated_file}")
```

##### SQLAlchemy Model Integration

Alternatively, you can use SQLAlchemy model classes directly as schema input:

```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
import random

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    department_id = Column(Integer, ForeignKey('departments.id'))

# Register a custom generator for foreign keys (will be auto-detected)
generator.register_generator('foreign_key', lambda row, col: random.choice([1, 2, 3]))

# Generate data directly from the model
df = generator.generate_data(
    schema=Employee,  # Pass the SQLAlchemy model directly!
    prompt="Generate employee data", 
    sample_size=10
)
```

- Foreign keys are automatically detected and assigned the type `'foreign_key'`
- Register a custom generator for foreign keys to maintain referential integrity
- Works with all SQLAlchemy column types and relationships

##### Output Options

- If `output_path` is provided (must end with `.csv` or `.json`), the file will be written and the method returns the file path.
- If not, a pandas DataFrame is returned.

See `examples/` directory for complete examples, including:
- Basic schema-based data generation
- SQLAlchemy model integration
- Foreign key relationship handling

## Optional REST API Service

### Prerequisites
- Python 3.8+
- PostgreSQL database
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd syda-service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Database Setup

1. Create a PostgreSQL database for the application.

2. Configure the database connection by creating a `.env` file in the root directory with the following variables:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/your_database_name
   ```

3. Run Alembic migrations to set up the database tables:
   ```bash
   # Navigate to the service directory
   cd service
   
   # Create initial migration (only needed once)
   alembic revision --autogenerate -m "Initial migration"
   
   # Apply migrations
   alembic upgrade head
   ```

### Running the API Service

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### Project Management
```
POST /projects
```
Create a new project
Example request:
```json
{
    "name": "my_project",
    "description": "My data masking project"
}
```

```
GET /projects
```
Get all projects

```
GET /projects/{project_name}/transactions
```
Get transactions for a specific project

#### Data Operations (Project-based)
```
POST /mask
```
Mask data for a project
Example request:
```json
{
    "project_name": "my_project",
    "data": {
        "email": ["test@example.com"],
        "phone": ["123-456-7890"]
    }
}
```

```
POST /generate
```
Generate synthetic data for a project

```
POST /generate/test-data
```
Generate test data for a project

```
POST /upload/mask
```
Upload and mask CSV for a project

```
POST /upload/generate
```
Upload and generate synthetic data for a project

```
POST /upload/unstructured
```
Process unstructured file (image, PDF, Word, Excel, text)

```
POST /upload/unstructured/generate
```
Generate synthetic data from unstructured file (Excel only)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
See the [LICENSE](LICENSE) file for the full license text.
