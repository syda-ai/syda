# Unstructured Document Generation

SYDA provides powerful capabilities for generating unstructured documents alongside structured data. This approach allows you to create realistic documents like invoices, contracts, reports, and more based on the structured data you generate.

## Document Template Basics

Document generation in SYDA is based on templates. You define a template that includes both static content and dynamic placeholders, which SYDA will fill with generated data.

## Key template attributes:

To generate documents, your schema must include special template attributes. 

| Attribute | Description | Example |
|-----------|-------------|---------|
| `__template__` | Whether this schema is a template | `true` |
| `__description__` | Human-readable description of the template | `Retail receipt template` |
| `__name__` | Name of the template | `Receipt` |
| `__depends_on__` | Other schemas this template depends on | `[Product, Transaction, Customer]` |
| `__foreign_keys__` | Field-level foreign key relationships | `customer_name: [Customer, first_name]` |
| `__template_source__` | Path to the template file | `templates/receipt.html` |
| `__input_file_type__` | Template format | `html` |
| `__output_file_type__` | Output document format | `pdf` |


Here's an example using YAML format:


```yaml
# receipt.yml
__template__: true
__description__: Retail receipt template
__name__: Receipt
__depends_on__: [Product, Transaction, Customer]
__foreign_keys__:
  customer_name: [Customer, first_name]
  customer_id: [Customer, id]
  
__template_source__: templates/receipt.html
__input_file_type__: html
__output_file_type__: pdf

# Regular schema fields
store_name:
  type: string
  length: 50
  description: Name of the retail store

store_address:
  type: address
  length: 150
  description: Full address of the store

store_phone:
  type: string
  length: 20
  description: Store phone number

receipt_number:
  type: string
  length: 12
  description: Unique receipt identifier

items:
  type: array
  description: List of purchased items with product details
```

Here's an example using SqlAlchemy model:


### SQLAlchemy Model-Based Templates

When using SQLAlchemy models, you can define template attributes directly as class attributes:

```python
import os
from sqlalchemy import Column, Integer, String, Float, Text, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')

class ContractDocument(Base):
    """Contract document for a won opportunity."""
    # Special metadata attributes
    __tablename__ = 'contract_documents'
    __depends_on__ = ['opportunities']
    
    # Template configuration as class attributes
    __template__ = True
    __template_source__ = os.path.join(templates_dir, 'contract.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False)
    effective_date = Column(Date)
    expiration_date = Column(Date)
    contract_number = Column(String(50))
    customer_name = Column(String(100), ForeignKey('customers.name'))
    customer_address = Column(String(200), ForeignKey('customers.address'))
    service_description = Column(Text)
    payment_terms = Column(Text)
    contract_value = Column(Float, ForeignKey('opportunities.value'))
    renewal_terms = Column(Text)
```

## Supported Template Formats

As of now SYDA supports HTML templates(Jinja2) for unstructured document generation.

### HTML Templates(Jinja2)

HTML Jinja2 templates provide the most flexibility and control over document formatting:

```html
{% raw %}
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Receipt #{{ receipt_number }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2cm; }
        .header { text-align: center; margin-bottom: 2em; }
        .receipt-details { margin-bottom: 2em; }
        .line-items { width: 100%; border-collapse: collapse; }
        .line-items th, .line-items td { border: 1px solid #ddd; padding: 8px; }
        .total { margin-top: 2em; text-align: right; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ store_name }}</h1>
        <p>{{ store_address }}</p>
        <p>{{ store_phone }}</p>
        <p>{{ store_website }}</p>
    </div>
    
    <div class="receipt-details">
        <p><strong>Receipt Number:</strong> {{ receipt_number }}</p>
        <p><strong>Date:</strong> {{ transaction_date }}</p>
        <p><strong>Time:</strong> {{ transaction_time }}</p>
        <p><strong>Customer:</strong> {{ customer_name }}</p>
        <p><strong>Customer ID:</strong> {{ customer_id }}</p>
    </div>
    
    <table class="line-items">
        <thead>
            <tr>
                <th>Item</th>
                <th>Quantity</th>
                <th>Unit Price</th>
                <th>Total</th>
            </tr>
        </thead>
        <tbody>
            {% if items %}
                {% for item in items %}
                <tr>
                    <td>{{ item.name }}</td>
                    <td>{{ item.quantity }}</td>
                    <td>${{ item.price }}</td>
                    <td>${{ item.total }}</td>
                </tr>
                {% endfor %}
            {% else %}
                <tr><td colspan="4">No items</td></tr>
            {% endif %}
        </tbody>
    </table>
    
    <div class="total">
        <p>Subtotal: ${{ subtotal }}</p>
        <p>Tax ({{ tax_rate }}%): ${{ tax_amount }}</p>
        <p>Discount: ${{ discount_amount }}</p>
        <p><strong>Total Amount: ${{ total }}</strong></p>
    </div>
</body>
</html>
{% endraw %}
```


## Template Design

SYDA uses Jinja2 for template rendering, providing powerful features for creating dynamic documents:

### Variables

Access any field from your schema directly:

```
{% raw %}
Customer: {{ customer_name }}
Invoice Number: {{ id }}
Amount Due: ${{ total_amount }}
{% endraw %}
```

### Loops

Iterate over arrays or lists of items:

```
{% raw %}
<table>
    <tr><th>Item</th><th>Price</th></tr>
    {% for item in items %}
    <tr>
        <td>{{ item.name }}</td>
        <td>${{ item.price }}</td>
    </tr>
    {% endfor %}
</table>
{% endraw %}
```

### Conditionals

Show or hide content based on conditions:

```
{% raw %}
{% if total_amount > 1000 %}
<div class="premium-customer">
    Thank you for your substantial order! You qualify for our premium support.
</div>
{% elif total_amount > 500 %}
<div class="valued-customer">
    Thank you for your order! You qualify for priority shipping.
</div>
{% else %}
<div class="standard-customer">
    Thank you for your order!
</div>
{% endif %}
{% endraw %}
```

### Filters

Transform data during rendering:

```
{% raw %}
Date: {{ issue_date | date_format('%B %d, %Y') }}
Name: {{ customer_name | upper }}
Summary: {{ description | truncate(100) }}
{% endraw %}
```



### Jinja2 Template Syntax Requirements

SYDA uses Jinja2 for template rendering. Be sure to follow these syntax requirements:

{% raw %}
* Use `{{ variable }}` for variable interpolation (with spaces inside the braces)
* Use `{% for item in items %}...{% endfor %}` for loops
* Use `{% if condition %}...{% endif %}` for conditionals
* Use `{# This is a comment #}` for comments
* Use `{{ variable | filter }}` for applying filters

**Important:** Do not use Handlebars-style syntax (e.g., `{{variable}}` without spaces or `{{\#each items}}`) as these won't be processed correctly.
{% endraw %}

#### Example of Correct Jinja2 Syntax:

```html
<div class="items">
  {% if items %}
    <h3>Items Purchased</h3>
    <table>
      {% for item in items %}
        <tr>
          <td>{{ item.name }}</td>
          <td>${{ item.price }}</td>
        </tr>
      {% endfor %}
    </table>
  {% else %}
    <p>No items purchased</p>
  {% endif %}
</div>
```

## PDF Generation

SYDA can automatically convert HTML and Markdown templates to PDF documents:

```python
schemas = {
    'Contract': {
        '__template__': 'templates/contract.html',
        '__template_source__': 'file',
        '__input_file_type__': 'html',
        '__output_file_type__': 'pdf',  # Generate PDF output
        
        'id': {'type': 'integer', 'primary_key': True},
        'client_name': {'type': 'string'},
        'start_date': {'type': 'date'},
        'end_date': {'type': 'date'},
        'contract_terms': {'type': 'string', 'format': 'long_text'}
    }
}

results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={'Contract': 5},
    output_dir='output/contracts'
)
```

This will generate a `Contract` directory containing PDF files (e.g., `Contract_1.pdf`, `Contract_2.pdf`, etc.)




## Best Practices

1. **Use HTML for Complex Layouts**: HTML provides the most control over document appearance
2. **Test Templates Separately**: Validate templates with sample data before full generation
3. **Include CSS in HTML Templates**: Embed CSS for consistent styling in PDF output
4. **Use Loops for Repetitive Content**: Generate tables, lists, and repeated sections efficiently
5. **Handle Optional Fields**: Use conditionals or defaults for fields that might be missing
6. **Consider Page Breaks**: For multi-page documents, control page breaks with CSS
7. **Document Variable Names**: Comment your templates to document expected variables



## Examples

To see unstructured document generation in action, explore  [SQLAlchemy Example](../examples/structured_and_unstructured_mixed/sqlalchemy_models.md) and [Yaml Example](../examples/structured_and_unstructured_mixed/yaml_schemas.md) 