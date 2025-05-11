
# Synthetic Data Generation Service -- WORK IN PROGRESS

A Python-based open-source service for generating synthetic data while preserving data utility.

## Features

### Core Features

- **Synthetic Data Generation**:
  - Statistical data generation
  - Pattern-based generation
  - Data distribution preservation
  - Synthetic data from various sources

### Optional Features

- **REST API Service**:
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

##### SQLAlchemy Model Integration with Referential Integrity

Alternatively, you can use SQLAlchemy model classes directly as schema input, including maintaining referential integrity between related models:

```python
from sqlalchemy import Column, Integer, String, ForeignKey, Float, create_engine
from sqlalchemy.orm import declarative_base, relationship
import random
import pandas as pd
from syda.structured import SyntheticDataGenerator

Base = declarative_base()

# Define related models with foreign key relationships
class Department(Base):
    __tablename__ = 'departments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    location = Column(String)
    budget = Column(Float)
    
    # One-to-many: one department has many employees
    employees = relationship("Employee", back_populates="department")
    
    def __repr__(self):
        return f"<Department(id={self.id}, name='{self.name}', location='{self.location}')>"


class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    department_id = Column(Integer, ForeignKey('departments.id'))
    role = Column(String)
    salary = Column(Float)
    
    # Many-to-one: many employees belong to one department
    department = relationship("Department", back_populates="employees")
    
    def __repr__(self):
        return f"<Employee(id={self.id}, name='{self.first_name} {self.last_name}', role='{self.role}')>"

# Step 1: Generate departments first
generator = SyntheticDataGenerator()
    
departments_df = generator.generate_data(
    schema=Department,
    prompt="""
    Generate realistic department data for a technology company.
    Departments should have names like Engineering, Marketing, Sales, HR, etc.
    Locations should be major cities around the world.
    Budget should be a realistic amount for each department, in USD.
    """,
    sample_size=5,
    output_path='departments.csv'
)

# Step 2: Create a custom foreign key generator for employees that references valid departments
departments_df = pd.read_csv('departments.csv')

# Register a custom generator for foreign key columns
def department_id_fk_generator(row, col_name):
    # Sample from the existing department IDs
    return random.choice(departments_df['id'].tolist())

# Register the custom foreign key generator
generator.register_generator('foreign_key', department_id_fk_generator)

# Step 3: Generate employee data with valid department_id references
employees_df = generator.generate_data(
    schema=Employee,
    prompt="""
    Generate realistic employee data for a technology company.
    Employees should have common first and last names.
    Emails should follow the pattern firstname.lastname@company.com.
    Roles should include software engineers, product managers, designers, etc.
    Salaries should be realistic amounts in USD.
    """,
    sample_size=20,
    output_path='employees.csv'
)

# Verify referential integrity
valid_dept_ids = set(departments_df['id'].tolist())
employee_dept_ids = set(employees_df['department_id'].tolist())

print("Verifying referential integrity...")
if employee_dept_ids.issubset(valid_dept_ids):
    print("✅ All employee department_id values reference valid departments")
else:
    invalid_ids = employee_dept_ids - valid_dept_ids
    print(f"❌ Found {len(invalid_ids)} invalid department_id references: {invalid_ids}")
```

Key features:
- Foreign keys are automatically detected and assigned the type `'foreign_key'`
- When handling related models, generate parent records first (departments)
- Register a custom generator for foreign keys that samples from existing valid IDs
- This approach maintains referential integrity across your generated data
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
    "description": "My synthetic data project"
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
