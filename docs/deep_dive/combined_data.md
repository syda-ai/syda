# Combined Structured and Unstructured Data

One of the most powerful features of SYDA is the ability to generate both structured database records and unstructured document content in a single integrated workflow. This capability is essential for many real-world applications that require consistency between database records and their associated documents.

## Connecting Documents to Structured Data

SYDA provides multiple methods for linking documents with their corresponding structured data:

### 1. Using Foreign Keys

Documents can reference structured data through foreign key relationships:

```python
# Dictionary schema example with foreign keys
schemas = {
    'Customer': {
        'id': {'type': 'integer', 'primary_key': True},
        'name': {'type': 'string'},
        'industry': {'type': 'string'},
    },
    'Contract': {
        '__template__': 'templates/contract.html',
        '__template_source__': 'file',
        '__input_file_type__': 'html',
        '__output_file_type__': 'pdf',
        
        'id': {'type': 'integer', 'primary_key': True},
        'title': {'type': 'string'},
        'customer_id': {
            'type': 'integer',
            'references': {'table': 'Customer', 'column': 'id'}
        },
        'terms': {'type': 'string'},
        'start_date': {'type': 'date'},
        'end_date': {'type': 'date'},
    }
}
```

### 2. Using SQLAlchemy Models with Templates

When using SQLAlchemy, you can define document models with template attributes:

```python
class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), comment="Name of the customer organization")
    industry = Column(String(50), comment="Customer's primary industry")

class Contract(Base):
    __tablename__ = 'contracts'
    
    # Template processing attributes
    __template__ = 'templates/contract_template.html'
    __template_source__ = 'file'
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), comment="Contract title")
    customer_id = Column(Integer, ForeignKey('customers.id'))
    terms = Column(Text, comment="Legal terms of the contract")
    start_date = Column(Date, comment="Contract start date")
    end_date = Column(Date, comment="Contract end date")
    
    # Define the relationship
    customer = relationship("Customer")
```

## Schema Dependencies for Documents

To ensure that document generation occurs after its referenced data is available, SYDA provides dependency management:

### 1. Explicit Dependencies

You can define explicit dependencies using the `__depends_on__` attribute:

```python
# In dictionary schemas
schemas = {
    'Customer': {
        'id': {'type': 'integer', 'primary_key': True},
        'name': {'type': 'string'},
    },
    'Contract': {
        '__depends_on__': ['Customer'],  # Explicit dependency
        '__template__': 'templates/contract.html',
        'id': {'type': 'integer', 'primary_key': True},
        'customer_id': {
            'type': 'integer',
            'references': {'table': 'Customer', 'column': 'id'}
        },
        'content': {'type': 'string'},
    }
}
```

### 2. Automatic Dependency Resolution

SYDA also automatically detects dependencies through foreign key relationships. This means that document schemas with foreign keys to other tables will be generated after those tables are available.

## Template Enrichment from Related Data

Document templates can access fields from related records using custom generators:

```python
def enrich_contract_data(table_name, column_name, row_data, dependencies=None):
    """
    Enriches contract data with customer information.
    """
    if not dependencies or 'Customer' not in dependencies:
        return row_data
    
    customer_id = row_data.get('customer_id')
    if customer_id is None:
        return row_data
    
    # Find the customer record with this ID
    customer = None
    for cust in dependencies['Customer']:
        if cust['id'] == customer_id:
            customer = cust
            break
    
    if customer:
        # Add customer fields to the contract data
        row_data['customer_name'] = customer['name']
        row_data['customer_industry'] = customer.get('industry', 'Unknown')
    
    return row_data
```

## Complete Example: CRM System with Documents

Here's a complete example of a CRM system that generates both structured data and documents:

```python
from syda import SyntheticDataGenerator, ModelConfig
from sqlalchemy import Column, Integer, String, Text, Date, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Define SQLAlchemy models
class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), comment="Customer company name")
    industry = Column(String(50), comment="Customer's industry")
    address = Column(String(200), comment="Customer's address")

class Opportunity(Base):
    __tablename__ = 'opportunities'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    name = Column(String(100), comment="Name/title of the opportunity")
    value = Column(Float, comment="Potential value of the opportunity")
    description = Column(Text, comment="Description of the opportunity")
    
    customer = relationship("Customer")

class ProposalDocument(Base):
    __tablename__ = 'proposal_documents'
    
    # Template attributes
    __template__ = 'templates/proposal_template.html'
    __template_source__ = 'file'
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    __depends_on__ = ['opportunities']
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'))
    title = Column(String(100), comment="Title of the proposal")
    subtitle = Column(String(200), comment="Subtitle or tagline")
    prepared_by = Column(String(100), comment="Name of the sales representative")
    created_date = Column(Date, comment="Date the proposal was created")
    proposed_solutions = Column(Text, comment="Detailed description of proposed solutions")
    implementation_timeline = Column(Text, comment="Timeline for implementation")
    pricing_details = Column(Text, comment="Pricing breakdown and details")
    terms_and_conditions = Column(Text, comment="Legal terms and conditions")
    
    opportunity = relationship("Opportunity")

# Custom generator to enrich proposals with customer and opportunity data
def enrich_proposal(table_name, column_name, row_data, dependencies=None):
    """Enrich proposal with customer and opportunity data for the template."""
    if not dependencies or 'opportunities' not in dependencies:
        return row_data
    
    opportunity_id = row_data.get('opportunity_id')
    if opportunity_id is None:
        return row_data
    
    # Find the opportunity
    opportunity = None
    for opp in dependencies['opportunities']:
        if opp['id'] == opportunity_id:
            opportunity = opp
            break
    
    if opportunity and 'customers' in dependencies:
        # Add opportunity data to proposal
        row_data['opportunity_name'] = opportunity['name']
        row_data['opportunity_value'] = opportunity['value']
        row_data['opportunity_description'] = opportunity['description']
        
        # Find and add customer data
        customer_id = opportunity['customer_id']
        for cust in dependencies['customers']:
            if cust['id'] == customer_id:
                row_data['customer_name'] = cust['name']
                row_data['customer_address'] = cust['address']
                break
    
    return row_data

# Generate data
def main():
    config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
    generator = SyntheticDataGenerator(model_config=config)
    
    custom_generators = {
        'proposal_documents': {
            '*': enrich_proposal  # Apply to all rows in the table
        }
    }
    
    # Generate all data with proper dependencies
    results = generator.generate_for_sqlalchemy_models(
        sqlalchemy_models=[Customer, Opportunity, ProposalDocument],
        sample_sizes={
            "customers": 5,
            "opportunities": 10,
            "proposal_documents": 10
        },
        prompts={
            "customers": "Generate diverse B2B technology customers",
            "opportunities": "Generate sales opportunities for enterprise software",
            "proposal_documents": "Generate professional sales proposals"
        },
        custom_generators=custom_generators,
        output_dir="output/crm"
    )
    
    print("Generated data:")
    for model_name, df in results.items():
        print(f"{model_name}: {len(df)} records")

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Define Clear Dependencies**: Use `__depends_on__` to ensure correct generation order
2. **Enrich Templates with Custom Generators**: Create custom generators that add fields from related tables
3. **Use Consistent Naming**: Maintain consistent field names between schemas and templates
4. **Handle Missing Data**: Ensure custom generators gracefully handle missing related records
5. **Optimize Template Performance**: Keep templates simple and efficient for large datasets
6. **Test End-to-End**: Always test the complete generation pipeline from database to documents
