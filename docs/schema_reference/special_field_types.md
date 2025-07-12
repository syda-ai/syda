# Special Field Types

Special field types are prefixed with double underscores. These special sections are validated during schema validation:

### `__description__`

It is used to identify the table description for the schema.

```yaml
__description__: Customer information for e-commerce site
```
### `__table_description__`

It can also be used to identify the table description for the schema.

```yaml
__table_description__: Customer information for e-commerce site
```

### `__foreign_keys__`

Defines foreign key relationships:

```yaml
__foreign_keys__:
  user_id: [User, id]
  product_id: [Product, id]
```

### `__depends_on__`

Specifies schema dependencies for generation order:

```yaml
__depends_on__: [Product, Customer]
```

This ensures that Product and Customer data are generated before the current schema.


## Special Template-Related Fields

For schemas that generate unstructured document outputs:

### `__template__`

It can be set to `true` or a string value to enable template generation.

```yaml
__template__: true
```

### `__template_source__`

It is used to specify the path to the template file.

```yaml
__template_source__: /path/to/template.html
```

### `__input_file_type__`

It is used to specify the input file type.

```yaml
__input_file_type__: html
```

### `__output_file_type__`

It is used to specify the output file type.

```yaml
__output_file_type__: pdf
```

These fields enable document generation from templates with the synthetic data.

> **Important**: When `__template__` is set to `true`, the `__template_source__` field is required. Schema validation will fail if this relationship is not maintained.