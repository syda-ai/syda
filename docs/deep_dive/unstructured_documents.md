# Unstructured Document Generation

SYDA provides powerful capabilities for generating unstructured documents alongside structured data. This approach allows you to create realistic documents like invoices, contracts, reports, and more based on the structured data you generate.

## Document Template Basics

Document generation in SYDA is based on templates. You define a template that includes both static content and dynamic placeholders, which SYDA will fill with generated data.

### Template Schema Requirements

To generate documents, your schema must include special template attributes:

```python
schemas = {
    'Invoice': {
        # Template attributes
        '__template__': 'templates/invoice.html',      # Path to template
        '__template_source__': 'file',                 # Source of template ('file' or 'schema')
        '__input_file_type__': 'html',                 # Template format
        '__output_file_type__': 'pdf',                 # Output format
        
        # Regular schema fields
        'id': {'type': 'integer', 'primary_key': True},
        'customer_name': {'type': 'string'},
        'issue_date': {'type': 'date'},
        'due_date': {'type': 'date'},
        'total_amount': {'type': 'number', 'format': 'float'},
        'line_items': {'type': 'array'}
    }
}
```

Key template attributes:

| Attribute | Description | Options |
|-----------|-------------|---------|
| `__template__` | Template path or content | File path or inline content |
| `__template_source__` | Where to find the template | 'file' or 'schema' |
| `__input_file_type__` | Template format | 'html', 'md', 'txt' |
| `__output_file_type__` | Output document format | 'pdf', 'html', 'txt' |
| `__output_filename_pattern__` | (Optional) Pattern for output filenames | e.g., 'invoice-{id}' |

## Supported Template Formats

SYDA supports multiple template formats:

### 1. HTML Templates (recommended)

HTML templates provide the most flexibility and control over document formatting:

```html
{% raw %}
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Invoice #{{ id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2cm; }
        .header { text-align: center; margin-bottom: 2em; }
        .invoice-details { margin-bottom: 2em; }
        .line-items { width: 100%; border-collapse: collapse; }
        .line-items th, .line-items td { border: 1px solid #ddd; padding: 8px; }
        .total { margin-top: 2em; text-align: right; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>INVOICE</h1>
        <h2>#{{ id }}</h2>
    </div>
    
    <div class="invoice-details">
        <p><strong>Customer:</strong> {{ customer_name }}</p>
        <p><strong>Issue Date:</strong> {{ issue_date }}</p>
        <p><strong>Due Date:</strong> {{ due_date }}</p>
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
            {% for item in line_items %}
            <tr>
                <td>{{ item.description }}</td>
                <td>{{ item.quantity }}</td>
                <td>${{ item.unit_price }}</td>
                <td>${{ item.quantity * item.unit_price }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div class="total">
        <p>Total Amount: ${{ total_amount }}</p>
    </div>
</body>
</html>
{% endraw %}
```

### 2. Markdown Templates

For simpler documents, Markdown templates can be used:

```markdown
{% raw %}
# Invoice #{{ id }}

**Customer:** {{ customer_name }}
**Issue Date:** {{ issue_date }}
**Due Date:** {{ due_date }}

## Line Items

| Item | Quantity | Unit Price | Total |
|------|----------|------------|-------|
{% for item in line_items %}
| {{ item.description }} | {{ item.quantity }} | ${{ item.unit_price }} | ${{ item.quantity * item.unit_price }} |
{% endfor %}

**Total Amount:** ${{ total_amount }}
{% endraw %}
```

### 3. Plain Text Templates

For the simplest documents, plain text templates are available:

```
{% raw %}
INVOICE #{{ id }}

Customer: {{ customer_name }}
Issue Date: {{ issue_date }}
Due Date: {{ due_date }}

Line Items:
{% for item in line_items %}
- {{ item.description }}: {{ item.quantity }} x ${{ item.unit_price }} = ${{ item.quantity * item.unit_price }}
{% endfor %}

Total Amount: ${{ total_amount }}
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

### Template Inheritance

For complex documents, you can use template inheritance:

```
{% raw %}
{% extends "base_template.html" %}

{% block content %}
    <h1>Invoice #{{ id }}</h1>
    <p>Customer: {{ customer_name }}</p>
{% endblock %}
{% endraw %}
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

This will generate:
1. A `Contract.csv` file with the structured data
2. A `Contract` directory containing PDF files (e.g., `Contract_1.pdf`, `Contract_2.pdf`, etc.)

## Template Source Options

SYDA supports two ways to provide templates:

### 1. File-Based Templates

Store templates in files and reference them by path:

```python
schemas = {
    'Report': {
        '__template__': 'templates/report.html',
        '__template_source__': 'file',  # Load from file
        # ... other fields
    }
}
```

### 2. Inline Templates

Define templates directly in your schema:

```python
schemas = {
    'SimpleReceipt': {
        '__template__': """
        <!DOCTYPE html>
        <html>
        <body>
            <h1>Receipt #{{ id }}</h1>
            <p>Amount: ${{ amount }}</p>
            <p>Date: {{ date }}</p>
        </body>
        </html>
        """,
        '__template_source__': 'schema',  # Template is inline
        '__input_file_type__': 'html',
        '__output_file_type__': 'pdf',
        
        'id': {'type': 'integer', 'primary_key': True},
        'amount': {'type': 'number', 'format': 'float'},
        'date': {'type': 'date'}
    }
}
```

## Working with SQLAlchemy Models

For SQLAlchemy models, define template attributes as class attributes:

```python
class ContractDocument(Base):
    __tablename__ = 'contract_documents'
    
    # Template attributes
    __template__ = 'templates/contract.html'
    __template_source__ = 'file'
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'))
    title = Column(String(100))
    content = Column(Text)
    signed_date = Column(Date)
    
    client = relationship("Client")
```

## Best Practices

1. **Use HTML for Complex Layouts**: HTML provides the most control over document appearance
2. **Test Templates Separately**: Validate templates with sample data before full generation
3. **Include CSS in HTML Templates**: Embed CSS for consistent styling in PDF output
4. **Use Loops for Repetitive Content**: Generate tables, lists, and repeated sections efficiently
5. **Handle Optional Fields**: Use conditionals or defaults for fields that might be missing
6. **Consider Page Breaks**: For multi-page documents, control page breaks with CSS
7. **Document Variable Names**: Comment your templates to document expected variables
8. **Use Raw Tags in Documentation**: When showing templates in Markdown documentation, wrap with `{% raw %}` and `{% endraw %}`
