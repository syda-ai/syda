#!/usr/bin/env python3
"""
Example demonstrating synthetic HR/Employee data generation using SYDA.
This example shows how to:
1. Generate related tables using YAML schemas
2. Respect foreign key relationships
3. Use custom generators for specific fields
4. Export results to CSV
"""

import os
import sys
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path so we can import syda
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from syda.generate import SyntheticDataGenerator
from syda.dependency_handler import ForeignKeyHandler

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(BASE_DIR, "schemas")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_email(row, col_name, **kwargs):
    """
    Custom generator for employee email based on first and last name.
    
    Args:
        row: The current row being processed 
        col_name: The name of the column being generated
    
    Returns:
        A realistic email address based on employee name
    """
    # Extract first and last name
    first_name = row.get('first_name', '')
    last_name = row.get('last_name', '')
    
    if not first_name or not last_name:
        return None
    
    # Normalize names: lowercase and remove spaces
    first = first_name.lower().replace(" ", "")
    last = last_name.lower().replace(" ", "")
    
    # Create email variations
    email_formats = [
        f"{first}.{last}@company.com",
        f"{first[0]}{last}@company.com",
        f"{first}{last[0]}@company.com",
        f"{last}.{first}@company.com"
    ]
    
    return random.choice(email_formats)

def calculate_manager_ids(row, col_name, parent_dfs=None):
    """
    Custom generator for department manager_id that ensures the manager
    belongs to the department they manage.
    
    Args:
        row: The current department row
        col_name: The column name being generated
        parent_dfs: Dictionary of generated dataframes
    
    Returns:
        A valid manager ID from the employees in the department
    """
    if not parent_dfs or 'Employee' not in parent_dfs:
        # If Employee data isn't available yet, return None
        return None
        
    dept_id = row.get('id')
    employees_df = parent_dfs['Employee']
    
    # Filter employees in this department
    dept_employees = employees_df[employees_df['department_id'] == dept_id]
    
    if len(dept_employees) == 0:
        return None
    
    # Select a manager from higher-level positions (level > 5)
    managers = dept_employees[dept_employees['position_id'] > 25]
    
    if len(managers) > 0:
        return int(random.choice(managers['id'].tolist()))
    else:
        # If no high-level positions, select any employee
        return int(random.choice(dept_employees['id'].tolist()))

def generate_realistic_salary(row, col_name, parent_dfs=None):
    """
    Custom generator to create realistic salaries based on position level.
    
    Args:
        row: The current employee row
        col_name: The column name being generated
        parent_dfs: Dictionary of generated dataframes
        
    Returns:
        A realistic salary value
    """
    if not parent_dfs or 'Position' not in parent_dfs:
        return random.randint(30000, 120000)
    
    position_id = row.get('position_id')
    positions_df = parent_dfs['Position']
    
    # Find the position
    position_row = positions_df[positions_df['id'] == position_id]
    if len(position_row) == 0:
        return random.randint(30000, 120000)
    
    # Get salary range
    min_salary = position_row.iloc[0]['min_salary']
    max_salary = position_row.iloc[0]['max_salary']
    
    # Generate a realistic salary within the range
    return round(random.uniform(min_salary, max_salary), -3)  # Round to nearest thousand

def generate_performance_review_dates(row, col_name, parent_dfs=None):
    """
    Custom generator for realistic review dates.
    
    Args:
        row: The current performance review row
        col_name: The column name being generated
        parent_dfs: Dictionary of previously generated dataframes
        
    Returns:
        A realistic review date
    """
    if not parent_dfs or 'Employee' not in parent_dfs:
        # Generate a random date in the past 2 years
        days_ago = random.randint(1, 730)  # Up to 2 years ago
        return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Get hire date for this employee
    employee_id = row.get('employee_id')
    employees_df = parent_dfs['Employee']
    employee_row = employees_df[employees_df['id'] == employee_id]
    
    if len(employee_row) == 0:
        # Employee not found, generate a random date
        days_ago = random.randint(1, 730)  # Up to 2 years ago
        return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Get hire date
    hire_date_str = employee_row.iloc[0]['hire_date']
    try:
        hire_date = datetime.strptime(hire_date_str, "%Y-%m-%d")
        
        # Generate a date between hire_date and now
        days_since_hire = (datetime.now() - hire_date).days
        if days_since_hire <= 0:
            return datetime.now().strftime("%Y-%m-%d")
        
        days_after_hire = random.randint(90, days_since_hire)  # At least 90 days after hire
        review_date = hire_date + timedelta(days=days_after_hire)
        return review_date.strftime("%Y-%m-%d")
    except:
        # Fallback if date parsing fails
        days_ago = random.randint(1, 730)  # Up to 2 years ago
        return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

def generate_review_periods(row, col_name, **kwargs):
    """
    Custom generator for review period start/end dates.
    
    Args:
        row: The current performance review row
        col_name: The column name being generated
        
    Returns:
        A start or end date for the review period
    """
    review_date_str = row.get('review_date')
    if not review_date_str:
        return None
        
    try:
        review_date = datetime.strptime(review_date_str, "%Y-%m-%d")
        
        if col_name == 'review_period_end':
            # End date is the review date
            return review_date.strftime("%Y-%m-%d")
        else:
            # Start date is typically 6 months to 1 year before end date
            days_before = random.randint(180, 365)
            start_date = review_date - timedelta(days=days_before)
            return start_date.strftime("%Y-%m-%d")
    except:
        return None

def main():
    print("HR/Employee Data Generation Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Define custom generators
    custom_generators = {
        'Employee': {
            'email': generate_email,
            'salary': generate_realistic_salary
        },
        'Department': {
            'manager_id': calculate_manager_ids
        },
        'PerformanceReview': {
            'review_date': generate_performance_review_dates,
            'review_period_start': generate_review_periods,
            'review_period_end': generate_review_periods
        }
    }
    
    # Define sample sizes
    sample_sizes = {
        'Department': 8,
        'Position': 15,
        'Employee': 50,
        'PerformanceReview': 75
    }
    
    # Define generation order to handle foreign key relationships correctly
    # (normally this would be handled automatically by the dependency handler)
    generation_order = ['Position', 'Department', 'Employee', 'PerformanceReview']
    
    print("\nGenerating synthetic HR data...")
    print("This example generates connected HR data across multiple tables")
    print("including departments, positions, employees, and performance reviews.")
    print("It uses custom generators to create realistic values and maintains")
    print("referential integrity across all tables.\n")
    
    # Generate data for all schemas
    # Load schemas from the schemas directory
    schemas = {}
    for schema_file in os.listdir(SCHEMA_DIR):
        if schema_file.endswith(('.yml', '.yaml')):
            schema_name = os.path.splitext(schema_file)[0].capitalize()
            schema_path = os.path.join(SCHEMA_DIR, schema_file)
            schemas[schema_name] = schema_path
    
    # Generate data for all schemas
    results = generator.generate_for_schemas(
        schemas=schemas,
        output_dir=OUTPUT_DIR,
        sample_sizes=sample_sizes,
        custom_generators=custom_generators,
        default_sample_size=10,
        default_prompt="Generate realistic HR data"
    )
    
    # Print summary of generated data
    print("\nGeneration Complete!")
    print("=" * 50)
    
    for schema_name, df in results.items():
        if df is not None:
            rows, cols = df.shape
            print(f"{schema_name}: {rows} rows x {cols} columns")
            
            # Print some sample data if available
            if rows > 0:
                if schema_name == 'Employee':
                    print("\nSample Employee Data:")
                    sample_cols = ['id', 'first_name', 'last_name', 'email', 'department_id', 'position_id']
                    print(df[sample_cols].head(3).to_string(index=False))
                    
                elif schema_name == 'Department':
                    print("\nSample Department Data:")
                    print(df.head(3).to_string(index=False))
    
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("\nYou can now analyze the generated data to see how foreign key")
    print("relationships are maintained and how custom generators create")
    print("more realistic values than pure-LLM generation.")
    
if __name__ == "__main__":
    main()
