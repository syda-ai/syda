# Output Options

SYDA offers flexible options for handling the output of generated data, allowing you to save results in various formats and locations.

## Return Types

By default, SYDA returns generated data as pandas DataFrames:

```python
from syda import SyntheticDataGenerator, ModelConfig

config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
generator = SyntheticDataGenerator(model_config=config)

# Generate data
results = generator.generate_for_schemas(
    schemas={...},
    sample_sizes={"Customer": 10}
)

# Results is a dictionary of DataFrames
customer_df = results["Customer"]

# Work with the DataFrame
print(f"Generated {len(customer_df)} customer records")
print(customer_df.head())
```

The returned `results` dictionary maps table names to pandas DataFrames, making it easy to analyze, transform, or further process the generated data.

## Saving to Files

You can save generated data to files by specifying an output directory:

```python
results = generator.generate_for_schemas(
    schemas={...},
    sample_sizes={"Customer": 10, "Order": 25},
    output_dir="output/crm_data"
)
```

When you provide an `output_dir`:

1. SYDA creates the directory if it doesn't exist
2. Each table's data is saved as a CSV file (e.g., `Customer.csv`, `Order.csv`)
3. The results dictionary still contains the DataFrames for immediate use

## Output Formats

By default, SYDA saves data in CSV format, but you can specify other formats using the `output_formats` parameter:

```python
results = generator.generate_for_schemas(
    schemas={...},
    sample_sizes={"Customer": 10, "Order": 25},
    output_dir="output/crm_data",
    output_formats=["csv", "json"]
)
```

Supported output formats include:

- `csv`: Standard comma-separated values format
- `json`: JSON format with records orientation

## Document Output

When generating unstructured documents alongside structured data, SYDA saves the documents in their specified formats:

```python
schemas = {
    'Report': {
        '__template__': 'true',
        '__template_source__': 'templates/report.html',
        '__input_file_type__': 'html',
        '__output_file_type__': 'pdf',
        # ...other fields
    }
}

results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={"Report": 5},
    output_dir="output/reports"
)
```

This creates:

- A `Report` subdirectory with the generated documents (e.g., `Report_1.pdf`, `Report_2.pdf`, etc.)

## Output Directory Structure

When using both structured data and document generation, SYDA creates an organized directory structure:

```
output/
├── Customer.csv
├── Order.csv
├── OrderItem.csv
├── Invoice/
│   ├── Invoice_1.pdf
│   ├── Invoice_2.pdf
│   └── ...
└── Report/
    ├── Report_1.pdf
    ├── Report_2.pdf
    └── ...
```

This structure makes it easy to locate and manage both structured data and generated documents.


## Working with Output Programmatically

After generation, you can further process or transform the output data:

```python
# Generate data
results = generator.generate_for_schemas(
    schemas={...},
    sample_sizes={"Customer": 10, "Order": 25}
)

# Process Customer data
customers = results["Customer"]
vip_customers = customers[customers["annual_revenue"] > 1000000]

# Process Order data
orders = results["Order"]
recent_orders = orders[orders["order_date"] > "2023-01-01"]

# Join data for analysis
merged = orders.merge(customers, left_on="customer_id", right_on="id")
```

## Best Practices

1. **Use Descriptive Output Directories**: Create meaningful directory names for your output
2. **Choose Appropriate Formats**: Select output formats based on your downstream needs
3. **Process DataFrames Before Saving**: Apply transformations before writing to disk when needed
4. **Check Output Size**: Be mindful of output size for large generations
5. **Backup Results**: Keep the returned DataFrames for immediate use even when saving to disk

## Examples

Explore  [SQLAlchemy Example](../examples/structured_and_unstructured_mixed/sqlalchemy_models.md) and [Yaml Example](../examples/structured_and_unstructured_mixed/yaml_schemas.md) 
