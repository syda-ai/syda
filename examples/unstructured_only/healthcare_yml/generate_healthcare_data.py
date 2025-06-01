#!/usr/bin/env python3
"""
Healthcare Unstructured Data Example

This script demonstrates how to generate synthetic healthcare data using YAML schemas
and template processing with PDF output.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the SYDA module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).parent
SCHEMAS_DIR = BASE_DIR / "schemas"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    """Main function to generate healthcare data."""
    print("=== Generating Healthcare Data with Templates ===")
    
    # Initialize generator with model config
    config = ModelConfig(provider="anthropic", model_name="claude-3-5-sonnet-20240620")
    generator = SyntheticDataGenerator(model_config=config)
    
    # Define schemas using YAML files
    schemas = {
        'MedicalReport': str(SCHEMAS_DIR / "medical_report.yml"),
        'LabResult': str(SCHEMAS_DIR / "lab_result.yml")
    }
    
    # Define sample sizes
    sample_sizes = {
        'MedicalReport': 5,
        'LabResult': 5
    }
    
    # Define custom prompts
    prompts = {
        'MedicalReport': 'Generate synthetic medical reports for patients with various health conditions',
        'LabResult': 'Generate synthetic laboratory test results for patients'
    }
    
    # Print information about template source paths
    for schema_name, schema_path in schemas.items():
        print(f"\n📄 Processing schema: {schema_name} from {schema_path}")
    
    # Generate data for all schemas at once
    print("\n🔄 Generating data for healthcare templates...")
    print("  The system will automatically determine the right generation order")
    print("  and handle template processing for PDF generation\n")
    
    results = generator.generate_for_schemas(
        schemas=schemas,
        sample_sizes=sample_sizes,
        prompts=prompts,
        output_dir=str(OUTPUT_DIR)
    )
    
    # Print summary of generated data
    print("\n✅ Data generation complete!")
    for schema_name, df in results.items():
        if df is not None:
            print(f"  {schema_name}: {len(df)} records")
            
            # Check for template output directories
            template_dir = OUTPUT_DIR / schema_name
            if template_dir.exists():
                files = list(template_dir.iterdir())
                pdf_files = [f for f in files if f.name.endswith('.pdf')]
                print(f"  - Found {len(pdf_files)} PDF files in {template_dir}")
                for i, pdf_file in enumerate(pdf_files[:3]):
                    print(f"    - {pdf_file.name}")
                if len(pdf_files) > 3:
                    print(f"    - ... and {len(pdf_files) - 3} more")
            else:
                print(f"  - Template directory for {schema_name} not found at {template_dir}")
    
    # Check for PDFs in all output directories
    print("\n🔍 Checking for generated PDF files:")
    total_pdfs = 0
    
    # Check main output directory first
    pdf_files = [f for f in OUTPUT_DIR.iterdir() if f.name.endswith('.pdf')]
    if pdf_files:
        total_pdfs += len(pdf_files)
        print(f"  - Found {len(pdf_files)} PDFs in main output directory")
    
    # Check schema-specific directories
    for schema_name in schemas:
        schema_output_dir = OUTPUT_DIR / schema_name
        if schema_output_dir.exists():
            pdf_files = [f for f in schema_output_dir.iterdir() if f.name.endswith('.pdf')]
            if pdf_files:
                total_pdfs += len(pdf_files)
                print(f"  - Found {len(pdf_files)} PDFs in {schema_name} directory")
    
    print(f"\n📊 Total PDFs generated: {total_pdfs}")
    print(f"📂 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
