# HR/Employee Data Generation Example

This example demonstrates how to use SYDA to generate synthetic HR/employee data with interlinked tables and custom generators.

## Overview

The HR example includes:

1. **Departments** - Company departments with budget and location information
2. **Positions** - Job positions with salary ranges and levels
3. **Employees** - Staff members linked to departments and positions
4. **Performance Reviews** - Regular employee evaluations with scores and feedback

This example showcases:
- Foreign key relationships across tables
- Custom generators for realistic data
- Generation dependencies to maintain data integrity
- Realistic business rules (managers from same department, salary ranges based on position)

## Schema Structure

The example defines four YAML schemas:

- `department.yml` - Department information with budget and location
- `position.yml` - Job positions with titles, levels, and salary ranges
- `employee.yml` - Employee records with links to departments and positions
- `performance_review.yml` - Employee reviews with scores and feedback

## Custom Generators

The example implements several custom generators:

1. **Email Generator** - Creates realistic emails based on employee names
2. **Manager ID Generator** - Ensures managers belong to departments they manage
3. **Salary Generator** - Sets salary within range defined by position
4. **Review Date Generator** - Generates review dates after employee hire dates
5. **Review Period Generator** - Creates sensible review periods

## Running the Example

To run this example:

```bash
cd /path/to/syda-fresh
python examples/structured_only/hr_employee_example/test_hr_schemas.py
```

The script will:
- Generate 8 departments, 15 positions, 50 employees, and 75 performance reviews
- Apply custom generators to create realistic values
- Maintain referential integrity across tables
- Save all data to CSV files in the `output` directory

## Output

The generated data is saved to the `output` directory as CSV files:
- `department.csv`
- `position.csv`
- `employee.csv`
- `performance_review.csv`

You can import these files into a database or analyze them to see the relationships between tables.
