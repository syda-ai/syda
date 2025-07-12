# Combined Structured and Unstructured Data

One of the most powerful features of SYDA is the ability to generate both structured database records and unstructured document content in a single integrated workflow. This capability is essential for many real-world applications that require consistency between database records and their associated documents.

## Connecting Documents to Structured Data

SYDA provides multiple methods for linking documents with their corresponding structured data:

### 1. Using Foreign Keys

Documents can reference structured data through foreign key relationships:

For example, a receipt document can reference a customer's name and id through a foreign key relationship:


```yaml
# Retail receipt template example
__template__: true
__description__: Retail receipt template
__name__: Receipt
__depends_on__: [Product, Transaction, Customer]
__foreign_keys__:
  customer_name: [Customer, first_name]
  customer_id: [Customer, id]
  
__template_source__: /templates/receipt.html
__input_file_type__: html
__output_file_type__: pdf

# Receipt header
store_name:
  type: string
  length: 50
  description: Name of the retail store

store_address:
  type: address
  length: 150
  description: Full address of the store

# Receipt details
receipt_number:
  type: string
  pattern: '^RCP-\d{8}$'
  length: 12
  description: Unique receipt identifier

transaction_date:
  type: date
  format: YYYY-MM-DD
  description: Date of the transaction

# Customer information
customer_name:
  type: string
  length: 100
  description: Full name of the customer

customer_id:
  type: integer
  description: Customer ID number

# Product purchase details
items:
  type: array
  description: "List of purchased items with product details"
  
# Totals
subtotal:
  type: float
  min: 0.99
  max: 999999.99
  decimals: 2
  description: Sum of all item totals before tax
```

### 2. Using SQLAlchemy Models with Templates

When using SQLAlchemy, you can define document models with template attributes:

For example, a contract document can reference an opportunity's name and id through a foreign key relationship:

```python
class ContractDocument(Base):
    """Contract document for a won opportunity."""
    # Special metadata attributes
    __tablename__ = 'contract_documents'
    __depends_on__ = ['opportunities']
    
    # Template configuration as regular fields (these become columns in the generated data)
    __template__ = True
    __template_source__ = os.path.join(templates_dir, 'contract.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True, comment='Primary key for contract document records')
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False, 
                           comment='Foreign key reference to the opportunity this contract is for')
    effective_date = Column(Date, comment='Date when the contract becomes effective')
    expiration_date = Column(Date, comment='Date when the contract expires')
    contract_number = Column(String(50), comment='Unique identifier/number for the contract')
    customer_name = Column(String(100), ForeignKey('customers.name'), 
                          comment='Name of the customer organization (linked to customers table)')
    customer_address = Column(String(200), ForeignKey('customers.address'), 
                             comment='Address of the customer organization (linked to customers table)')
    service_description = Column(Text, comment='Detailed description of services to be provided')
    payment_terms = Column(Text, comment='Payment terms including schedule, methods and conditions')
    contract_value = Column(Float, ForeignKey('opportunities.value'), 
                           comment='Total monetary value of the contract in USD (linked to opportunities table)')
    renewal_terms = Column(Text, comment='Terms for contract renewal or extension')
    legal_terms = Column(Text, comment='Legal terms and conditions including liabilities, warranties, etc.')

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
# In YAML schemas (from retail_yml/schemas/receipt.yml)
__template__: true
__description__: Retail receipt template
__name__: Receipt
__depends_on__: [Product, Transaction, Customer]  # Explicit dependencies
__foreign_keys__:
  customer_name: [Customer, first_name]
  customer_id: [Customer, id]
  
__template_source__: /templates/receipt.html
__input_file_type__: html
__output_file_type__: pdf

# In SQLAlchemy models (from crm_sqlalchemy/models.py)
class ContractDocument(Base):
    """Contract document for a won opportunity."""
    # Special metadata attributes
    __tablename__ = 'contract_documents'
    __depends_on__ = ['opportunities']  # Explicit dependency
    
    # Template configuration
    __template__ = True
    __template_source__ = os.path.join(templates_dir, 'contract.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False)
    customer_name = Column(String(100), ForeignKey('customers.name'))
    customer_address = Column(String(200), ForeignKey('customers.address'))
```

### 2. Implicit Dependencies via Foreign Keys

SYDA automatically detects dependencies through foreign key relationships. This means that document schemas with foreign keys to other tables will be generated after those tables are available.

For example, in the retail YAML example, SYDA detects that Receipt depends on Customer through the foreign keys:

## Template Enrichment from Related Data

Document templates can access fields from related records using custom generators:

```python
def enrich_contract_data(row, col_name=None, parent_dfs=None):
    """
    Enriches contract data with customer information.
    
    Args:
        row: The current row being processed (as a pandas Series or dict-like object)
        col_name: The name of the column being generated
        parent_dfs: Dictionary of previously generated dataframes (schema name as key)
    """
    if parent_dfs is None or 'Customer' not in parent_dfs:
        return row
    
    customer_id = row.get('customer_id')
    if customer_id is None:
        return row
    
    # Find the customer record with this ID using pandas filtering
    customers_df = parent_dfs['Customer']
    matching_customers = customers_df[customers_df['id'] == customer_id]
    
    if len(matching_customers) > 0:
        # Get the first matching customer
        customer = matching_customers.iloc[0]
    
    if len(matching_customers) > 0:
        # Add customer fields to the contract data
        row['customer_name'] = customer['name']
        row['customer_industry'] = customer.get('industry', 'Unknown')
    
    return row
```


## Best Practices

1. **Define Clear Dependencies**: Use `__depends_on__` to ensure correct generation order
2. **Enrich Templates with Custom Generators**: Create custom generators that add fields from related tables
3. **Use Consistent Naming**: Maintain consistent field names between schemas and templates
4. **Optimize Template Performance**: Keep templates simple and efficient for large datasets
5. **Define Foreign Keys Properly**: SYDA supports 2 methods for defining foreign keys:
   - Using `__foreign_keys__` special section in schemas
   - Using field-level `references` properties within type definitions


## Examples

To see combined data in action, explore  [SQLAlchemy Example](../examples/structured_and_unstructured_mixed/sqlalchemy_models.md) and [Yaml Example](../examples/structured_and_unstructured_mixed/yaml_schemas.md) 