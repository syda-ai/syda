#!/usr/bin/env python
"""
Example of using SyntheticDataGenerator with multiple foreign key relationships.
This example demonstrates:
1. Creating SQLAlchemy models with multiple foreign key relationships
2. Registering column-specific generators for different foreign key columns
3. Generating synthetic data while maintaining referential integrity
"""

import sys
import os
import random
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, ForeignKey, Float, create_engine
from sqlalchemy.orm import declarative_base, relationship

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the synthetic data generator
from syda.structured import SyntheticDataGenerator

# Create a Base for our models
Base = declarative_base()

# Define our models
class Department(Base):
    """Organizational department that groups employees and oversees projects.
    Departments have specific business functions and geographic locations.
    """
    __tablename__ = 'departments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True, 
                 comment="Official department name (e.g. Engineering, Marketing)")
    location = Column(String(100), comment="Primary office location of this department")
    
    # One-to-many: one department has many employees and projects
    employees = relationship("Employee", back_populates="department")
    projects = relationship("Project", back_populates="department")
    
    def __repr__(self):
        return f"<Department(id={self.id}, name='{self.name}', location='{self.location}')>"


class Employee(Base):
    """Company employee with personal and professional details.
    Employees belong to a department and may manage multiple projects.
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
    
    # Many-to-one: many employees belong to one department
    department = relationship("Department", back_populates="employees")
    # One-to-many: one employee manages many projects
    managed_projects = relationship("Project", back_populates="manager")
    
    def __repr__(self):
        return f"<Employee(id={self.id}, name='{self.first_name} {self.last_name}')>"


class Project(Base):
    """Business project with details on ownership, management, and resources.
    Each project belongs to a department and is managed by a specific employee.
    """
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, 
                 comment="Project title or name")
    department_id = Column(Integer, ForeignKey('departments.id'),
                          comment="Department responsible for this project")
    manager_id = Column(Integer, ForeignKey('employees.id'),
                       comment="Employee who manages this project")
    budget = Column(Float, comment="Total budget allocated for the project in USD")
    
    # Many-to-one: many projects belong to one department
    department = relationship("Department", back_populates="projects")
    # Many-to-one: many projects are managed by one employee
    manager = relationship("Employee", back_populates="managed_projects")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', budget={self.budget})>"


def generate_department_data():
    """Generate synthetic department data."""
    # Get API key from environment or from .env file
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ Warning: OPENAI_API_KEY not found in environment variables. Make sure it's set in your .env file.")
    
    # Create a model config with explicit settings for better reliability
    from syda.schemas import ModelConfig
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",  # Using 3.5 for faster/cheaper responses
        temperature=0.7,  
        max_tokens=4000  # Using higher max_tokens value
    )
    
    # Initialize with specific model config
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # The prompt describes what kind of data we want
    # Note: Most details are now extracted from the model metadata
    prompt = """
    Generate realistic department data for a technology company.
    Include 5 departments covering Engineering, Marketing, Sales, HR, and Finance.
    Each department should have a unique ID and a specific location.
    Ensure data is properly structured and complete.
    """
    
    # Generate data directly from the Department model
    output_path = 'departments.csv'
    generator.generate_data(
        sqlalchemy_model=Department,
        prompt=prompt,
        sample_size=5,
        output_path=output_path
    )
    
    print(f"Department data written to {output_path}")
    # Return the data as a dataframe for the next step
    return pd.read_csv(output_path)


def generate_employee_data(departments_df):
    """Generate synthetic employee data with valid department_id foreign keys."""
    # Use the same model config for consistency
    from syda.schemas import ModelConfig
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=4000  # Using higher max_tokens value
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Register a column-specific generator for department_id foreign key
    def department_id_generator(row, col_name):
        # Sample from the generated department IDs
        return random.choice(departments_df['id'].tolist())
    
    generator.register_generator('foreign_key', department_id_generator, column_name='department_id')
    
    # More specific prompt with clear instructions
    prompt = """
    Generate realistic employee data for a technology company.
    Include a mix of different roles, salaries, and hire dates.
    Assign employees to existing departments.
    Ensure each employee has a valid department_id from the departments table.
    Include senior and junior employees with appropriate titles and salaries.
    """
    
    # Generate data directly from the Employee model
    output_path = 'employees.csv'
    generator.generate_data(
        sqlalchemy_model=Employee,
        prompt=prompt,
        sample_size=20,  # More employees than departments
        output_path=output_path
    )
    
    print(f"Employee data written to {output_path}")
    return pd.read_csv(output_path)


def generate_project_data(departments_df, employees_df):
    """Generate synthetic project data with department and manager references."""
    # Use the same model config for consistency
    from syda.schemas import ModelConfig
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=4000  # Using higher max_tokens value
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Register custom generators for the foreign keys
    # 1. Department ID generator
    def department_id_generator(row, col_name):
        return random.choice(departments_df['id'].tolist())
    
    # 2. Manager ID generator - only use employee IDs for managers
    def manager_id_generator(row, col_name):
        return random.choice(employees_df['id'].tolist())
    
    # Register our foreign key generators for specific columns
    generator.register_generator('foreign_key', department_id_generator, column_name='department_id')
    generator.register_generator('foreign_key', manager_id_generator, column_name='manager_id')
    
    # More specific prompt with clear instructions
    prompt = """
    Generate realistic project data for a technology company.
    Include a mix of different statuses, budgets, and start/end dates.
    Ensure each project has a valid department_id and manager_id.
    Projects should have realistic names, descriptions, and appropriate budgets.
    Include both ongoing and completed projects with realistic timelines.
    """
    
    # Generate data directly from the Project model
    output_path = 'projects.csv'
    generator.generate_data(
        sqlalchemy_model=Project,
        prompt=prompt,
        sample_size=8,  # Fewer projects than departments
        output_path=output_path
    )
    
    print(f"Project data written to {output_path}")
    return pd.read_csv(output_path)


def verify_referential_integrity(departments_df, employees_df, projects_df):
    """Verify that all foreign keys reference valid primary keys."""
    valid_dept_ids = set(departments_df['id'].tolist())
    valid_emp_ids = set(employees_df['id'].tolist())
    
    # Check department_id in employees
    emp_dept_ids = set(employees_df['department_id'].tolist())
    if emp_dept_ids.issubset(valid_dept_ids):
        print("✅ All employee department_id values reference valid departments")
    else:
        invalid_ids = emp_dept_ids - valid_dept_ids
        print(f"❌ Found {len(invalid_ids)} invalid employee department_id references: {invalid_ids}")
    
    # Check department_id in projects
    proj_dept_ids = set(projects_df['department_id'].tolist())
    if proj_dept_ids.issubset(valid_dept_ids):
        print("✅ All project department_id values reference valid departments")
    else:
        invalid_ids = proj_dept_ids - valid_dept_ids
        print(f"❌ Found {len(invalid_ids)} invalid project department_id references: {invalid_ids}")
    
    # Check manager_id in projects
    proj_mgr_ids = set(projects_df['manager_id'].tolist())
    if proj_mgr_ids.issubset(valid_emp_ids):
        print("✅ All project manager_id values reference valid employees")
    else:
        invalid_ids = proj_mgr_ids - valid_emp_ids
        print(f"❌ Found {len(invalid_ids)} invalid project manager_id references: {invalid_ids}")


def main():
    """Run the demo."""
    
    print("Generating synthetic department data...")
    departments_df = generate_department_data()
    print(departments_df)
    
    print("\nGenerating synthetic employee data...")
    employees_df = generate_employee_data(departments_df)
    print(employees_df)
    
    print("\nGenerating synthetic project data with multiple foreign keys...")
    projects_df = generate_project_data(departments_df, employees_df)
    print(projects_df)
    
    # Verify referential integrity for all relationships
    print("\nVerifying referential integrity...")
    verify_referential_integrity(departments_df, employees_df, projects_df)


if __name__ == "__main__":
    main()
