
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

##### SQLAlchemy Model Integration with SQLAlchemy, Smart Metadata Extraction & Foreign Key Constraints

You can use SQLAlchemy model classes directly as schema input with enhanced metadata extraction that leverages model docstrings and column comments:

```python
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship
import random
import pandas as pd
from syda.structured import SyntheticDataGenerator

Base = declarative_base()

# Define models with rich docstrings and metadata
class Department(Base):
    """Organizational department that groups employees by business function.
    Departments are the primary organizational units within the company structure.
    """
    __tablename__ = 'departments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True, 
                 comment="Official department name (e.g. Engineering, Marketing)")
    location = Column(String(100), comment="Primary office location of this department")
    budget = Column(Float, comment="Annual budget allocation in USD")
    
    # One-to-many: one department has many employees
    employees = relationship("Employee", back_populates="department")


class Employee(Base):
    """Company employee with their personal and professional details.
    Each employee belongs to a specific department and has associated job information.
    """
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    first_name = Column(String(50), nullable=False, 
                       comment="Employee's given name")
    last_name = Column(String(50), nullable=False, 
                      comment="Employee's family name")
    email = Column(String(100), nullable=False, unique=True, 
                  comment="Business email address for communication")
    department_id = Column(Integer, ForeignKey('departments.id'),
                          comment="Department where employee works")
    role = Column(String(100), comment="Job title or position within the company")
    salary = Column(Float, comment="Annual salary in USD")
    
    # Many-to-one: many employees belong to one department
    department = relationship("Department", back_populates="employees")

# Step 1: Generate departments first
generator = SyntheticDataGenerator()
    
# The system automatically extracts docstrings and column metadata
# so the prompt can be much simpler
departments_df = generator.generate_data(
    schema=Department,
    prompt="Generate realistic department data for a technology company.",
    sample_size=5,
    output_path='departments.csv'
)

# Step 2: Create custom generators for specific columns
departments_df = pd.read_csv('departments.csv')

# Register a column-specific generator for department_id
def department_id_fk_generator(row, col_name):
    # Sample from the existing department IDs
    return random.choice(departments_df['id'].tolist())

# Register the custom generator specifically for the department_id column
generator.register_generator('foreign_key', department_id_fk_generator, column_name='department_id')

# Step 3: Generate employee data with valid department_id references
# The system uses model docstrings and column metadata to inform the generation
employees_df = generator.generate_data(
    schema=Employee,
    prompt="Generate realistic employee data for a technology company.",
    sample_size=10,
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

##### Handling Foreign Key Relationships

The library provides robust support for handling foreign key relationships with referential integrity:

1. **Automatic Foreign Key Detection**: Foreign keys are automatically detected from your SQLAlchemy models and assigned the type `'foreign_key'`.

2. **Column-Specific Foreign Key Generators**: Register different generators for each foreign key column when dealing with multiple relationships:

```python
# Different generators for different foreign key columns
generator.register_generator('foreign_key', department_id_generator, column_name='department_id')
generator.register_generator('foreign_key', manager_id_generator, column_name='manager_id')
```

3. **Multi-Step Generation Process**: For related tables, generate parent records first, then use their IDs when generating child records:

```python
# Generate departments first
departments_df = generator.generate_data(schema=Department, ...)

# Then generate employees with department references
employees_df = generator.generate_data(schema=Employee, ...)

# Finally generate projects with both department and employee references
projects_df = generator.generate_data(schema=Project, ...)
```

4. **Referential Integrity Preservation**: The foreign key generator samples from actual existing IDs in the parent table, ensuring all references are valid.

5. **Metadata-Enhanced Foreign Keys**: Column comments on foreign key fields are preserved and included in the prompt, helping the LLM understand the relationship context.

##### Metadata Enhancement Benefits

This approach provides several powerful benefits:

1. **Richer Context for LLMs**: The system extracts docstrings, comments, constraints, and other metadata from your SQLAlchemy models, providing rich context that helps the LLM generate more accurate and domain-appropriate synthetic data.

2. **Simpler Prompts**: You can write much shorter prompts because the system automatically includes essential information from your model definitions.

3. **Constraint-Aware Generation**: The LLM becomes aware of constraints like `nullable=False`, `unique=True`, and field lengths, leading to data that better respects your schema rules.

4. **Column-Specific Generators**: You can register generators for specific columns (not just data types), allowing precise control over how each field is populated.

5. **Automatic Docstring Utilization**: Class and field docstrings are automatically incorporated into the generation process, ensuring the LLM understands the business context of your models.

Under the hood, the system constructs a detailed prompt that includes all relevant metadata:

```
Model Description: Company employee with their personal and professional details...

Generate 10 records JSON objects with these fields:
- id: number (primary_key)
- first_name: text (not_null) - Employee's given name
- last_name: text (not_null) - Employee's family name
- email: text (unique, not_null) - Business email address for communication
- department_id: foreign_key - Department where employee works
...
```

This allows for more accurate and contextually appropriate synthetic data generation while requiring less manual specification in your code.

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
