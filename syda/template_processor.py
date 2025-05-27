"""
Template processing functionality for converting synthetic data to documents.
"""

import os
import pandas as pd
import logging
import jinja2
from weasyprint import HTML

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_templates(template_schemas, structured_results, sample_sizes, default_sample_size=10, output_dir=None):
    """
    Process template schemas to generate documents.
    
    Args:
        template_schemas: Dictionary mapping template names to schema definitions
        structured_results: Dictionary mapping schema names to DataFrames of generated data
        sample_sizes: Dictionary mapping schema names to desired sample sizes
        default_sample_size: Default sample size if not specified in sample_sizes
        output_dir: Directory to save generated documents
        
    Returns:
        Dictionary mapping template schema names to DataFrames of template data
    """
    results = {}
    
    # Process each template schema
    for schema_name, schema_def in template_schemas.items():
        logger.info(f"Processing template schema: {schema_name}")
        
        # Get the DataFrame with template data
        if schema_name not in structured_results:
            logger.error(f"No data found for template schema {schema_name}")
            continue
            
        df = structured_results[schema_name]
        results[schema_name] = df
        
        # Skip document generation if no output directory
        if output_dir is None:
            logger.info(f"Skipping document generation for {schema_name} (no output directory specified)")
            continue
            
        # Create output subdirectory for this template
        template_output_dir = os.path.join(output_dir, schema_name)
        os.makedirs(template_output_dir, exist_ok=True)
        
        # Process each row to generate a document
        documents_generated = 0
        
        logger.info(f"Processing {schema_name} templates...")
        for idx, row in df.iterrows():
            doc_num = idx + 1
            logger.info(f"Processing template {doc_num}/{len(df)}...")
            
            # Get template details
            template_source = row.get('template_source')
            input_type = row.get('input_file_type', '').lower()
            output_type = row.get('output_file_type', '').lower()
            
            # Skip if missing required fields
            if not template_source or not input_type or not output_type:
                logger.error(f"Template {doc_num} missing required fields")
                continue
                
            # Verify template file exists
            if not os.path.exists(template_source):
                logger.error(f"Template source not found: {template_source}")
                continue
                
            # Output path for this document
            output_path = os.path.join(template_output_dir, f"document_{doc_num}.{output_type}")
            
            logger.info(f"  - Template source: {template_source}")
            logger.info(f"  - Output path: {output_path}")
            
            try:
                # Process based on input type
                if input_type == 'html':
                    # Print detailed debug info
                    logger.info(f"DEBUG: Processing HTML template: {template_source}")
                    logger.info(f"DEBUG: Template file exists: {os.path.exists(template_source)}")
                    logger.info(f"DEBUG: Template dir: {os.path.dirname(template_source)}")
                    logger.info(f"DEBUG: Template file: {os.path.basename(template_source)}")
                    
                    # Convert row to dict for template processing
                    context = row.to_dict()
                    logger.info(f"DEBUG: Context keys: {list(context.keys())[:5]}...")
                    
                    # Render the template with Jinja2
                    template_dir = os.path.dirname(template_source)
                    template_file = os.path.basename(template_source)
                    logger.info(f"DEBUG: Setting up Jinja2 with dir: {template_dir} and file: {template_file}")
                    
                    # Read template file directly to verify it exists and has content
                    try:
                        with open(template_source, 'r') as f:
                            template_content = f.read()
                            logger.info(f"DEBUG: Template file content length: {len(template_content)} characters")
                    except Exception as file_error:
                        logger.error(f"DEBUG: Error reading template file: {str(file_error)}")
                        raise
                    
                    # Set up Jinja environment
                    env = jinja2.Environment(
                        loader=jinja2.FileSystemLoader(template_dir),
                        autoescape=jinja2.select_autoescape(['html', 'xml'])
                    )
                    
                    try:
                        template = env.get_template(template_file)
                        logger.info("DEBUG: Successfully loaded Jinja2 template")
                    except Exception as jinja_error:
                        logger.error(f"DEBUG: Error loading Jinja2 template: {str(jinja_error)}")
                        raise
                    
                    try:
                        rendered_html = template.render(**context)
                        logger.info(f"DEBUG: Successfully rendered HTML: {len(rendered_html)} characters")
                    except Exception as render_error:
                        logger.error(f"DEBUG: Error rendering template: {str(render_error)}")
                        raise
                    
                    # Generate PDF from HTML
                    if output_type == 'pdf':
                        try:
                            logger.info(f"DEBUG: Generating PDF with WeasyPrint")
                            HTML(string=rendered_html).write_pdf(output_path)
                            logger.info(f"DEBUG: PDF file generated: {os.path.exists(output_path)}")
                        except Exception as pdf_error:
                            logger.error(f"DEBUG: Error generating PDF: {str(pdf_error)}")
                            raise
                    else:
                        # Write rendered HTML to file
                        with open(output_path, 'w') as f:
                            f.write(rendered_html)
                            
                    documents_generated += 1
                    logger.info(f"✓ Successfully generated: {output_path}")
                else:
                    logger.error(f"Unsupported input type: {input_type}")
                    continue
            except Exception as e:
                logger.error(f"Error generating document {doc_num} for {schema_name}: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
                
        logger.info(f"Generated {documents_generated} documents for {schema_name}")
    
    return results
