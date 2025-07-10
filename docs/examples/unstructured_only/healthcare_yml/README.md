# Healthcare Unstructured Data Example

This example demonstrates how to use SYDA for generating unstructured healthcare data using YAML schemas with PDF output.

## Overview

This example shows how to:

1. Define healthcare data schemas with template metadata using YAML format
2. Reference HTML templates for document formatting
3. Generate synthetic healthcare documents in PDF format using the `generate_for_schemas` method
4. Process schema definitions with proper template integration

## Directory Structure

- `schemas/`: Contains YAML schema definitions with template metadata
  - `medical_report.yml`: Schema for medical reports with template settings
  - `lab_result.yml`: Schema for lab results with template settings
- `templates/`: HTML templates for medical reports and lab results
  - `medical_report_template.html`: Template for medical visit reports
  - `lab_result_template.html`: Template for laboratory test results
- `output/`: Generated healthcare data files (PDF format and JSON data)
- `generate_healthcare_data.py`: Main script to generate healthcare data using schemas

## Running the Example

Execute the example script:

```bash
python3 examples/unstructured_only/healthcare_yml/generate_healthcare_data.py
```

This will:
- Initialize the `SyntheticDataGenerator` with model configuration
- Load the YAML schema files with template references
- Generate 5 synthetic medical reports and 5 lab results as PDF files
- Save the synthetic data in both PDF format and structured JSON data

## Schema Definitions

The example uses YAML files that define both the schema and template metadata:

1. **Medical Report** (`medical_report.yml`): Contains:
   - Template metadata (`__template__`, `__name__`, etc.)
   - Input/output format configuration (HTML to PDF)
   - Data fields for patient information, vital signs, diagnosis, etc.

2. **Lab Result** (`lab_result.yml`): Contains:
   - Template metadata and format configuration
   - Fields for laboratory test information, results, and reference ranges

## Templates

HTML templates in the `templates/` directory define how medical reports and lab results are formatted. The schema files reference these templates and specify conversion to PDF format using the following special attributes:

```yaml
__template__: true
__template_source: path/to/template.html
__input_file_type__: html
__output_file_type__: pdf
```

## Using the Results

After generation, the example produces:

- PDF documents that can be viewed in any PDF reader
- Structured JSON data representing the synthetic healthcare records

These can be used for:
- Training machine learning models
- Testing document processing pipelines
- Demonstrating healthcare data workflows
- Prototyping healthcare applications
