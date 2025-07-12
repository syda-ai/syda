# SQLAlchemy Models with Mixed Content

> Source code: [examples/structured_and_unstructured/crm_sqlalchemy/example_crm_templates.py](https://github.com/syda-ai/syda/blob/main/examples/structured_and_unstructured/crm_sqlalchemy/test_crm_templates.py)

This example demonstrates how to use SQLAlchemy models to generate both structured database data and unstructured document content for a CRM system.

## Overview

SYDA can generate both structured tabular data and unstructured document content in a single workflow. This approach is particularly valuable in business systems where documents (like proposals and contracts) need to consistently reference database records (like customers and opportunities).

In this CRM example, we generate:
- Structured data: Customers, contacts, and sales opportunities
- Unstructured content: Proposal documents and contract documents as PDFs

## Model Definition

The CRM example uses SQLAlchemy models to define both structured database tables and document templates.

### Structured Data Models

Here are the SQLAlchemy models for the structured database tables:

```python
class Customer(Base):
    """Customer organization in the CRM system."""
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True, comment='Primary key for customer records')
    name = Column(String(100), nullable=False, comment='Full name of the customer organization')
    industry = Column(String(50), comment='Industry sector the customer operates in')
    annual_revenue = Column(Float, comment='Annual revenue of the customer organization in USD')
    employees = Column(Integer, comment='Total number of employees in the customer organization')
    website = Column(String(100), comment='Website URL of the customer organization')
    address = Column(String(200), comment='Physical street address of the customer headquarters')
    city = Column(String(50), comment='City where the customer headquarters is located')
    state = Column(String(2), comment='Two-letter state/province code')
    zip_code = Column(String(10), comment='Postal/zip code of the customer headquarters')
    status = Column(String(20), comment='Current status of the customer relationship (Active, Inactive, Prospect)')
    
    # Relationships
    contacts = relationship("Contact", back_populates="customer")
    opportunities = relationship("Opportunity", back_populates="customer")


class Contact(Base):
    """Individual person associated with a customer."""
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True, comment='Primary key for contact records')
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, comment='Foreign key reference to the customer this contact belongs to')
    first_name = Column(String(50), nullable=False, comment='Contact\'s first/given name')
    last_name = Column(String(50), nullable=False, comment='Contact\'s last/family name')
    email = Column(String(100), nullable=False, unique=True, comment='Contact\'s email address (unique across all contacts)')
    phone = Column(String(20), comment='Contact\'s phone number including country/area code')
    position = Column(String(100), comment='Job title or role within the customer organization')
    is_primary = Column(Boolean, comment='Whether this is the primary point of contact for the customer')
    
    # Relationships
    customer = relationship("Customer", back_populates="contacts")


class Opportunity(Base):
    """Sales opportunity with a customer."""
    __tablename__ = 'opportunities'
    
    id = Column(Integer, primary_key=True, comment='Primary key for opportunity records')
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, comment='Foreign key reference to the customer this opportunity is with')
    name = Column(String(100), nullable=False, comment='Name or title of the sales opportunity')
    value = Column(Float, nullable=False, comment='Estimated monetary value of the opportunity in USD')
    stage = Column(String(20), nullable=False, comment='Current stage in the sales pipeline (Lead, Qualification, Proposal, Negotiation, Closed Won, Closed Lost)')
    probability = Column(Float, comment='Estimated probability (0-100%) of winning the opportunity')
    expected_close_date = Column(Date, comment='Expected date when the opportunity will be closed (won or lost)')
    description = Column(Text, comment='Detailed description of the opportunity and customer requirements')
    
    # Relationships
    customer = relationship("Customer", back_populates="opportunities")
```

### Document Template Models

For generating unstructured documents (proposals and contracts), we define special SQLAlchemy models with template processing capabilities:

```python
class ProposalDocument(Base):
    """Sales proposal document for an opportunity."""
    # Special metadata attributes
    __tablename__ = 'proposal_documents'
    __depends_on__ = ['opportunities']
    
    # Template configuration as regular fields
    __template__ = True
    __template_source__ = os.path.join(templates_dir, 'proposal.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True, comment='Primary key for proposal document records')
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False, comment='Foreign key reference to the opportunity this proposal is for')
    created_date = Column(Date, comment='Date when the proposal document was created')
    title = Column(String(200), comment='Main title of the proposal document')
    subtitle = Column(String(300), comment='Secondary title or tagline for the proposal')
    prepared_by = Column(String(100), comment='Name of the sales representative who prepared the proposal')
    customer_name = Column(String(100), ForeignKey('customers.name'), comment='Name of the customer organization (linked to customers table)')
    customer_address = Column(String(200), ForeignKey('customers.address'), comment='Address of the customer organization (linked to customers table)')
    opportunity_name = Column(String(100), ForeignKey('opportunities.name'), comment='Name of the opportunity (linked to opportunities table)')
    opportunity_value = Column(Float, ForeignKey('opportunities.value'), comment='Value of the opportunity in USD (linked to opportunities table)')
    opportunity_description = Column(Text, ForeignKey('opportunities.description'), comment='Description of the opportunity (linked to opportunities table)')
    proposed_solutions = Column(Text, comment='Detailed description of the proposed solutions/products/services')
    implementation_timeline = Column(Text, comment='Timeline for implementing the proposed solutions')
    pricing_details = Column(Text, comment='Detailed pricing information, including breakdowns and options')
    terms_and_conditions = Column(Text, comment='Standard terms and conditions for the proposal')


class ContractDocument(Base):
    """Contract document for a won opportunity."""
    # Special metadata attributes
    __tablename__ = 'contract_documents'
    __depends_on__ = ['opportunities']
    
    # Template configuration as regular fields
    __template__ = True
    __template_source__ = os.path.join(templates_dir, 'contract.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True, comment='Primary key for contract document records')
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False, comment='Foreign key reference to the opportunity this contract is for')
    effective_date = Column(Date, comment='Date when the contract becomes effective')
    expiration_date = Column(Date, comment='Date when the contract expires')
    contract_number = Column(String(50), comment='Unique identifier/number for the contract')
    customer_name = Column(String(100), ForeignKey('customers.name'), comment='Name of the customer organization (linked to customers table)')
    customer_address = Column(String(200), ForeignKey('customers.address'), comment='Address of the customer organization (linked to customers table)')
    service_description = Column(Text, comment='Detailed description of services to be provided')
    payment_terms = Column(Text, comment='Payment terms including schedule, methods and conditions')
    contract_value = Column(Float, ForeignKey('opportunities.value'), comment='Total monetary value of the contract in USD (linked to opportunities table)')
    renewal_terms = Column(Text, comment='Terms for contract renewal or extension')
    legal_terms = Column(Text, comment='Legal terms and conditions including liabilities, warranties, etc.')
    signatures = Column(Text, comment='Signature blocks for both parties')
```

## Template Processing Attributes

For template-based document generation, SQLAlchemy models use special class attributes:

1. **`__template__`**: Set to `True` to indicate this model generates document content
2. **`__template_source__`**: Path to the HTML template file
3. **`__input_file_type__`**: Format of the source template (`html`)
4. **`__output_file_type__`**: Format of the generated document (`pdf`)
5. **`__depends_on__`**: List of other models that must be generated first

These attributes tell SYDA to:
1. Generate structured data for this model
2. Use the template to format the data into a document
3. Convert the formatted document to PDF

## Foreign Key Handling

SQLAlchemy models define foreign keys explicitly through `ForeignKey()` definitions, which SYDA uses to detect relationships:

```python
customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
```

For template models, you can also define dependencies using the `__depends_on__` attribute:

```python
__depends_on__ = ['opportunities']
```

This ensures that required data (like opportunities) is generated before processing templates.

## Code Example

Here's how to use SQLAlchemy models to generate both structured data and document content:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig
import os
import models  # Import SQLAlchemy model definitions

def main():
    """Main entry point for the example."""
    
    # Initialize generator with model config
    config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
    generator = SyntheticDataGenerator(model_config=config)
    
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data for all models in one call
    results = generator.generate_for_sqlalchemy_models(
        sqlalchemy_models=[
            models.Customer,
            models.Contact,
            models.Opportunity,
            models.ProposalDocument, 
            models.ContractDocument
        ],
        sample_sizes={
            'customers': 5,
            'contacts': 10,
            'opportunities': 8,
            'proposal_documents': 3,
            'contract_documents': 2
        },
        prompts={
            'customers': "Generate a customer for the opportunity",
            'contacts': "Generate a contact for the customer",
            'opportunities': "Generate an opportunity for the customer",
            'proposal_documents': "Generate a proposal document for the opportunity",
            'contract_documents': "Generate a contract document for the opportunity"
        },
        output_dir=output_dir
    )
    
    # Print summary of generated data
    print("\n✅ Data generation complete!")
    for model_name, df in results.items():
        print(f"  {model_name}: {len(df)} records")
```

## Document Templates

The template models reference HTML template files. Here's a simplified version of what these templates might contain:

### Proposal Template

```html
{% raw %}
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{title}}</title>
    <style>
        body { font-family: 'Arial', sans-serif; line-height: 1.6; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin: 20px 0; }
        .footer { margin-top: 50px; font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{title}}</h1>
        <h2>{{subtitle}}</h2>
        <p>Prepared by: {{prepared_by}}<br>
        Created: {{created_date}}</p>
    </div>
    
    <div class="section">
        <h2>Customer Information</h2>
        <p>
            <strong>{{customer_name}}</strong><br>
            {{customer_address}}
        </p>
    </div>
    
    <div class="section">
        <h2>Opportunity: {{opportunity_name}}</h2>
        <p><strong>Value:</strong> ${{opportunity_value}}</p>
        <p>{{opportunity_description}}</p>
    </div>
    
    <div class="section">
        <h2>Proposed Solutions</h2>
        <p>{{proposed_solutions}}</p>
    </div>
    
    <div class="section">
        <h2>Implementation Timeline</h2>
        <p>{{implementation_timeline}}</p>
    </div>
    
    <div class="section">
        <h2>Pricing Details</h2>
        <p>{{pricing_details}}</p>
    </div>
    
    <div class="section">
        <h2>Terms and Conditions</h2>
        <p>{{terms_and_conditions}}</p>
    </div>
</body>
</html>
{% endraw %}
```

## Key Features

1. **Mixed Content Generation**: Generate both structured data and document PDFs in one workflow
2. **Template-Based Documents**: Convert structured data into formatted PDF documents
3. **Data Consistency**: Ensure generated documents reference valid database records
4. **Referential Integrity**: Maintain proper relationships between database tables and documents
5. **Dependency Management**: Define which models must be generated before others

## Best Practices

1. **Model Documentation**: Add clear docstrings to describe each model's purpose
   ```python
   """Sales proposal document for an opportunity."""
   ```

2. **Column Comments**: Use column comments to guide data generation
   ```python
   comment='Name of the sales representative who prepared the proposal'
   ```

3. **Explicit Dependencies**: Define `__depends_on__` to ensure proper generation order
   ```python
   __depends_on__ = ['opportunities']
   ```

4. **ForeignKey Definitions**: Define explicit foreign key relationships
   ```python
   ForeignKey('opportunities.id')
   ```

5. **Template Structure**: Create clean HTML templates with appropriate sections and styling

## Sample Outputs

You can view sample outputs generated using these SQLAlchemy models and templates here:

> [Example CRM SQLAlchemy Model Outputs](https://github.com/syda-ai/syda/tree/main/examples/structured_and_unstructured/crm_sqlalchemy/output)
