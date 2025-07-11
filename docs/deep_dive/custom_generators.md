# Custom Generators

Custom generators in SYDA provide a powerful way to control and customize the data generation process. They allow you to define specific logic for generating or transforming field values beyond what the LLM can do alone.

## When to Use Custom Generators

Custom generators are particularly useful when you need to:

1. Generate specialized domain-specific data
2. Ensure consistency across related records
3. Compute values based on other fields
4. Implement complex business rules
5. Access data from related tables
6. Post-process LLM-generated content

## Creating Custom Generators

A custom generator is simply a Python function that follows this signature:

```python
def my_custom_generator(table_name, column_name, row_data, dependencies=None):
    """
    Generate or transform a value for a specific field.
    
    Args:
        table_name (str): Name of the table being generated
        column_name (str): Name of the column being generated
        row_data (dict): Current row data with values generated so far
        dependencies (dict, optional): Generated data from dependencies
        
    Returns:
        The value to use for this field
    """
    # Your custom logic here
    return generated_value
```

### Simple Example: Calculating a Value

```python
def calculate_total(table_name, column_name, row_data, dependencies=None):
    """Calculate the total price based on quantity and unit_price."""
    if 'quantity' in row_data and 'unit_price' in row_data:
        return round(row_data['quantity'] * row_data['unit_price'], 2)
    return None
```

### Complex Example: Generating Domain-Specific Data

```python
import random
from datetime import datetime, timedelta

def generate_tracking_number(table_name, column_name, row_data, dependencies=None):
    """Generate a realistic package tracking number."""
    carriers = {
        'UPS': lambda: f"1Z{random.randint(100000000, 999999999)}",
        'FedEx': lambda: f"{random.randint(1000000000, 9999999999)}",
        'USPS': lambda: f"9{random.randint(4000000000000000, 4999999999999999)}"
    }
    
    carrier = row_data.get('shipping_carrier', random.choice(list(carriers.keys())))
    if carrier in carriers:
        return carriers[carrier]()
    else:
        # Default format if carrier not recognized
        return f"TRK{random.randint(10000000, 99999999)}"
```

## Registering Custom Generators

Custom generators are registered by passing them to the generator function:

```python
from syda import SyntheticDataGenerator, ModelConfig

# Define custom generators
def calculate_total(table_name, column_name, row_data, dependencies=None):
    if 'quantity' in row_data and 'unit_price' in row_data:
        return round(row_data['quantity'] * row_data['unit_price'], 2)
    return None

def calculate_tax(table_name, column_name, row_data, dependencies=None):
    if 'total_amount' in row_data:
        return round(row_data['total_amount'] * 0.0725, 2)  # 7.25% tax rate
    return None

# Register custom generators
custom_generators = {
    'Order': {  # Table name
        'total_amount': calculate_total,  # Column-specific generator
        'tax_amount': calculate_tax
    }
}

# Use custom generators during data generation
config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)
results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={'Order': 10},
    custom_generators=custom_generators
)
```

## Wildcard Custom Generators

You can also define a generator that applies to all fields in a table using the `*` wildcard:

```python
def log_and_pass_through(table_name, column_name, row_data, dependencies=None):
    """Log all field generation and pass through the original value."""
    print(f"Generating {column_name} for {table_name}: {row_data.get(column_name)}")
    return row_data.get(column_name)

custom_generators = {
    'Customer': {
        '*': log_and_pass_through  # Applied to all fields in Customer table
    }
}
```

## Accessing Related Tables

One of the most powerful features of custom generators is the ability to access data from related tables:

```python
def enrich_invoice_with_customer_data(table_name, column_name, row_data, dependencies=None):
    """Add customer information to invoice description."""
    if not dependencies or 'Customer' not in dependencies:
        return f"Invoice #{row_data.get('id', 'Unknown')}"
        
    customer_id = row_data.get('customer_id')
    if customer_id is None:
        return f"Invoice #{row_data.get('id', 'Unknown')}"
        
    # Find the customer with matching ID
    customer = None
    for cust in dependencies['Customer']:
        if cust['id'] == customer_id:
            customer = cust
            break
            
    if customer:
        return f"Invoice #{row_data.get('id', 'Unknown')} - {customer['name']}"
    else:
        return f"Invoice #{row_data.get('id', 'Unknown')}"
```

## Generator Execution Order

SYDA executes generators in a specific order:

1. LLM generates initial values for all fields without custom generators
2. Custom generators are applied in the order fields are defined in the schema
3. For each field with a custom generator:
   - The generator receives the current state of the row
   - Any fields already processed are available
   - The generator can modify any value, not just its target field

## Example Use Cases

### 1. Calculating Derived Values

```python
def calculate_bmi(table_name, column_name, row_data, dependencies=None):
    """Calculate BMI from height and weight."""
    height_cm = row_data.get('height_cm')
    weight_kg = row_data.get('weight_kg')
    
    if height_cm and weight_kg:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m * height_m)
        return round(bmi, 1)
    return None
```

### 2. Generating Realistic Financial Data

```python
import random
from datetime import datetime, timedelta

def generate_transaction_data(table_name, column_name, row_data, dependencies=None):
    """Generate realistic financial transaction data."""
    if column_name == 'transaction_date':
        # Generate dates within the last 30 days
        days_ago = random.randint(0, 30)
        return (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
    elif column_name == 'amount':
        # Generate realistic transaction amounts
        transaction_type = row_data.get('transaction_type', 'purchase')
        
        if transaction_type == 'purchase':
            # Most purchases between $5-100, some larger
            if random.random() < 0.9:
                return round(random.uniform(5, 100), 2)
            else:
                return round(random.uniform(100, 1000), 2)
                
        elif transaction_type == 'deposit':
            # Deposits typically larger
            return round(random.uniform(100, 5000), 2)
            
        elif transaction_type == 'withdrawal':
            # Withdrawals similar to purchases
            return round(random.uniform(20, 300), 2)
            
        return round(random.uniform(5, 100), 2)
```

### 3. Foreign Key Distribution Control

```python
import random

def assign_managers(table_name, column_name, row_data, dependencies=None):
    """
    Distribute employees to managers following a realistic pattern:
    - 20% employees have no manager (executives)
    - 80% are distributed among existing employees
    """
    if random.random() < 0.2:
        return None  # 20% have no manager (top level)
    
    if not dependencies or 'employees' not in dependencies:
        return None
        
    # Find all potential managers (employees created so far)
    # Exclude the current employee being generated
    potential_managers = [
        e['id'] for e in dependencies['employees'] 
        if e['id'] != row_data.get('id')
    ]
    
    if not potential_managers:
        return None
        
    # Select a manager randomly
    return random.choice(potential_managers)
```

### 4. Document Content Generation

```python
def generate_document_content(table_name, column_name, row_data, dependencies=None):
    """
    Generate appropriate document content based on document type.
    This is used for fields that will be rendered in templates.
    """
    document_type = row_data.get('document_type', 'generic')
    
    if document_type == 'invoice':
        return f"""
        Thank you for your purchase on {row_data.get('issue_date', 'recent date')}.
        
        Your invoice total is ${row_data.get('total_amount', 0):.2f}.
        
        Payment is due by {row_data.get('due_date', 'future date')}.
        """
    
    elif document_type == 'contract':
        return f"""
        This agreement, dated {row_data.get('start_date', 'today')}, 
        is between {row_data.get('provider_name', 'Provider')} and
        {row_data.get('client_name', 'Client')}.
        
        The term of this agreement is from {row_data.get('start_date', 'start')}
        to {row_data.get('end_date', 'end')}.
        """
    
    else:
        return "Generic document content."
```

## Best Practices

1. **Keep Generators Simple**: Each generator should have a single responsibility
2. **Handle Missing Data**: Always check if required fields exist before using them
3. **Use Type Hints**: Add type hints to make your generators more maintainable
4. **Document Behavior**: Add docstrings explaining what each generator does
5. **Error Handling**: Include error handling to prevent failures
6. **Test in Isolation**: Test generators independently with sample data
7. **Avoid Side Effects**: Generators should not modify external state
8. **Consider Performance**: For large datasets, optimize generator performance

## Advanced Generator Patterns

### Chain of Responsibility

```python
def apply_discount(table_name, column_name, row_data, dependencies=None):
    """Apply discounts based on various business rules."""
    if column_name != 'final_price':
        return row_data.get(column_name)
        
    base_price = row_data.get('price', 0)
    discount = 0
    
    # Apply membership discount
    if row_data.get('customer_type') == 'member':
        discount += base_price * 0.05  # 5% member discount
        
    # Apply quantity discount
    quantity = row_data.get('quantity', 1)
    if quantity >= 10:
        discount += base_price * 0.1  # 10% bulk discount
    elif quantity >= 5:
        discount += base_price * 0.05  # 5% smaller bulk discount
        
    # Apply seasonal promotion
    if row_data.get('is_promotional_period', False):
        discount += base_price * 0.15  # 15% promotional discount
        
    # Ensure discount doesn't exceed 30% of the base price
    discount = min(discount, base_price * 0.3)
    
    return round(base_price - discount, 2)
```

### Factory Pattern

```python
def address_generator_factory(country):
    """Factory function that returns a country-specific address generator."""
    
    def us_address_generator(table_name, column_name, row_data, dependencies=None):
        """Generate US-formatted addresses."""
        # US-specific address generation logic
        return {
            'street': f"{random.randint(1, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Washington'])} St",
            'city': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston']),
            'state': random.choice(['NY', 'CA', 'IL', 'TX']),
            'zip': f"{random.randint(10000, 99999)}",
            'country': 'USA'
        }
        
    def uk_address_generator(table_name, column_name, row_data, dependencies=None):
        """Generate UK-formatted addresses."""
        # UK-specific address generation logic
        return {
            'street': f"{random.randint(1, 100)} {random.choice(['High', 'Church', 'Park', 'London'])} Road",
            'city': random.choice(['London', 'Manchester', 'Birmingham', 'Liverpool']),
            'postcode': f"{random.choice(['AB', 'CD', 'EF', 'GH'])}{random.randint(1, 99)} {random.randint(1, 9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",
            'country': 'UK'
        }
    
    # Return the appropriate generator based on country
    if country.lower() == 'uk':
        return uk_address_generator
    else:
        return us_address_generator  # Default to US

# Usage
custom_generators = {
    'Customer': {
        'address': address_generator_factory('US')
    }
}
```
