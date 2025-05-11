#!/usr/bin/env python
"""
Example of using SyntheticDataGenerator with SQLAlchemy models.
This example demonstrates:
1. Creating simple SQLAlchemy models with foreign key relationships
2. Generating synthetic data directly from these models
3. Handling foreign keys with custom generators
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


def generate_department_data():
    """Generate synthetic department data."""
    generator = SyntheticDataGenerator()
    
    # The prompt describes what kind of data we want
    prompt = """
    Generate realistic department data for a technology company.
    Departments should have names like Engineering, Marketing, Sales, HR, etc.
    Locations should be major cities around the world.
    Budget should be a realistic amount for each department, in USD.
    """
    
    # Generate data directly from the Department model
    output_path = 'departments.csv'
    generator.generate_data(
        schema=Department,  # Pass the SQLAlchemy model directly
        prompt=prompt,
        sample_size=5,
        output_path=output_path
    )
    
    print(f"Department data written to {output_path}")
    # Return the data as a dataframe for the next step
    return pd.read_csv(output_path)


def generate_employee_data(departments_df):
    """Generate synthetic employee data with valid department_id foreign keys."""
    generator = SyntheticDataGenerator()
    
    # Register a custom generator for foreign key columns
    # This will sample from existing department IDs
    def department_id_fk_generator(row, col_name):
        # Sample from the generated department IDs
        return random.choice(departments_df['id'].tolist())
    
    # Register our custom foreign key generator
    generator.register_generator('foreign_key', department_id_fk_generator)
    
    # The prompt describes what kind of data we want
    prompt = """
    Generate realistic employee data for a technology company.
    Employees should have common first and last names.
    Emails should follow the pattern firstname.lastname@company.com.
    Roles should include software engineers, product managers, designers, and other tech roles.
    Salaries should be realistic amounts in USD.
    """
    
    # Generate data directly from the Employee model
    output_path = 'employees.csv'
    generator.generate_data(
        schema=Employee,  # Pass the SQLAlchemy model directly
        prompt=prompt,
        sample_size=20,
        output_path=output_path
    )
    
    print(f"Employee data written to {output_path}")
    return pd.read_csv(output_path)


def main():
    """Run the demo."""
    
    print("Generating synthetic department data...")
    departments_df = generate_department_data()
    print(departments_df)
    
    print("\nGenerating synthetic employee data with foreign keys...")
    employees_df = generate_employee_data(departments_df)
    print(employees_df)
    
    # Verify referential integrity
    valid_dept_ids = set(departments_df['id'].tolist())
    employee_dept_ids = set(employees_df['department_id'].tolist())
    
    print("\nVerifying referential integrity...")
    if employee_dept_ids.issubset(valid_dept_ids):
        print("✅ All employee department_id values reference valid departments")
    else:
        invalid_ids = employee_dept_ids - valid_dept_ids
        print(f"❌ Found {len(invalid_ids)} invalid department_id references: {invalid_ids}")


if __name__ == "__main__":
    main()
