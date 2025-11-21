# Schema Validation Fix: Foreign Keys & Jinja Templates

## Issue Summary
**Title:** Before generating data with AI, add validators to check schema fields have valid foreign key relations and Jinja templates placeholders are present in the schema

**Problem:**
Currently, SYDA generates synthetic data without pre-validation of critical schema elements:
1. **Silent Foreign Key Failures**: If a foreign key references a non-existent table/column, the data generation proceeds and silently fails or produces corrupted data
2. **Missing Template Placeholders**: If a Jinja template schema references fields that don't exist in the template file, generation fails mid-process
3. **Invalid Constraint Definitions**: Schema definitions with malformed constraints are not caught until runtime

**Impact:**
- Data integrity violations
- Confusing error messages deep in the generation pipeline
- Wasted computation on invalid schemas
- Corrupted datasets that pass initial checks

---

## Solution Architecture

### 1. **Schema Validation Framework**

Create a new validation module: `syda/validators.py` that performs pre-generation validation with three components:

```
┌─────────────────────────────────────────────────────────────┐
│              Schema Validation Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. FOREIGN KEY VALIDATOR                             │  │
│  │    - Verify referenced tables exist                  │  │
│  │    - Verify referenced columns exist                 │  │
│  │    - Detect circular dependencies                    │  │
│  │    - Check naming convention inference               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. TEMPLATE PLACEHOLDER VALIDATOR                    │  │
│  │    - Extract all {{field}} placeholders              │  │
│  │    - Verify each placeholder exists in schema        │  │
│  │    - Check Jinja2 syntax validity                    │  │
│  │    - Validate template file paths                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. CONSTRAINT VALIDATOR                              │  │
│  │    - Validate numeric constraints (min/max)          │  │
│  │    - Validate string constraints (pattern, length)   │  │
│  │    - Check constraint value ranges                   │  │
│  │    - Validate custom generator signatures           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. SCHEMA INTEGRATION VALIDATOR                       │  │
│  │    - Run all validators in sequence                  │  │
│  │    - Collect and report all errors                   │  │
│  │    - Suggest fixes for common issues                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### **STEP 1: Create `syda/validators.py`**

**Purpose:** Centralized validation logic for all schema types

**Key Classes:**
- `SchemaValidator`: Main validation orchestrator
- `ForeignKeyValidator`: FK-specific validation
- `TemplateValidator`: Template placeholder validation
- `ConstraintValidator`: Field constraint validation
- `ValidationError`: Custom exception with detailed error info

**Pseudo-code:**
```python
class ValidationError(Exception):
    """Raised when schema validation fails"""
    def __init__(self, schema_name: str, errors: List[str]):
        self.schema_name = schema_name
        self.errors = errors  # List of error messages
        self.error_count = len(errors)

class SchemaValidator:
    """Main schema validation orchestrator"""
    
    def validate_schemas(self, schemas, schema_metadata) -> ValidationResult:
        """
        Validate all schemas before generation
        
        Returns:
            ValidationResult with status, errors, warnings, suggestions
        """
        results = {}
        
        for schema_name, schema in schemas.items():
            # 1. Validate foreign keys
            fk_errors = self._validate_foreign_keys(
                schema_name, 
                schema, 
                schemas  # Pass all schemas for cross-reference
            )
            
            # 2. Validate templates
            template_errors = self._validate_templates(schema_name, schema)
            
            # 3. Validate constraints
            constraint_errors = self._validate_constraints(schema_name, schema)
            
            # Aggregate errors
            results[schema_name] = {
                'foreign_key_errors': fk_errors,
                'template_errors': template_errors,
                'constraint_errors': constraint_errors,
                'is_valid': not (fk_errors or template_errors or constraint_errors)
            }
        
        return ValidationResult(results)

class ForeignKeyValidator:
    """Validate foreign key relationships"""
    
    def validate_foreign_keys(self, schema_name: str, schema: Dict, 
                            all_schemas: Dict[str, Dict]) -> List[str]:
        """
        Validate all foreign keys in a schema
        
        Checks:
        1. Referenced schema exists in all_schemas
        2. Referenced column exists in target schema
        3. Naming convention inference is correct (with warnings)
        4. No circular dependencies
        5. Field type is compatible with foreign key
        """
        errors = []
        
        fks = schema.get('__foreign_keys__', {})
        for fk_field, (target_schema, target_column) in fks.items():
            # Check target schema exists
            if target_schema not in all_schemas:
                errors.append(
                    f"FK: Field '{fk_field}' references non-existent schema '{target_schema}'"
                )
                continue
            
            # Check target column exists
            target_schema_def = all_schemas[target_schema]
            if target_column not in target_schema_def:
                errors.append(
                    f"FK: Field '{fk_field}' references non-existent column "
                    f"'{target_schema}.{target_column}'"
                )
            
            # Check for naming convention issues (e.g., user_id -> users.id)
            if not self._is_naming_convention_valid(fk_field, target_schema):
                errors.append(
                    f"FK: Field '{fk_field}' has potentially invalid naming convention "
                    f"for schema '{target_schema}' (expected '*_{self._singularize(target_schema)}_id')"
                )
        
        return errors

class TemplateValidator:
    """Validate template placeholders and Jinja2 syntax"""
    
    def validate_templates(self, schema_name: str, schema: Dict) -> List[str]:
        """
        Validate template-related fields
        
        Checks:
        1. Template file exists
        2. All {{placeholders}} reference existing schema fields
        3. Jinja2 syntax is valid
        4. Required template metadata is present
        """
        errors = []
        
        # Check if this is a template schema
        if '__template_source__' not in schema:
            return errors  # Not a template schema
        
        template_path = schema['__template_source__']
        
        # Check file exists
        if not os.path.exists(template_path):
            errors.append(
                f"Template: File not found: '{template_path}'"
            )
            return errors
        
        # Extract placeholders from template file
        try:
            content = self._read_template_file(template_path)
            placeholders = self._extract_jinja_placeholders(content)
            
            # Validate each placeholder
            for placeholder in placeholders:
                if placeholder not in schema:
                    errors.append(
                        f"Template: Placeholder '{{{{ {placeholder} }}}}' is not defined in schema"
                    )
            
            # Check required metadata
            if '__input_file_type__' not in schema:
                errors.append("Template: Missing '__input_file_type__' metadata")
            if '__output_file_type__' not in schema:
                errors.append("Template: Missing '__output_file_type__' metadata")
                
        except Exception as e:
            errors.append(f"Template: Error reading template: {str(e)}")
        
        return errors

class ConstraintValidator:
    """Validate field constraints"""
    
    def validate_constraints(self, schema_name: str, schema: Dict) -> List[str]:
        """
        Validate field constraints
        
        Checks:
        1. min <= max for numeric constraints
        2. String patterns are valid regex
        3. Constraint values are of correct type
        4. Custom generators are callable
        """
        errors = []
        
        for field_name, field_def in schema.items():
            if field_name.startswith('__'):
                continue  # Skip metadata fields
            
            if isinstance(field_def, dict) and 'constraints' in field_def:
                constraints = field_def['constraints']
                
                # Numeric constraint validation
                if 'min' in constraints and 'max' in constraints:
                    if constraints['min'] > constraints['max']:
                        errors.append(
                            f"Constraint: Field '{field_name}' has min > max "
                            f"({constraints['min']} > {constraints['max']})"
                        )
                
                # String pattern validation
                if 'pattern' in constraints:
                    try:
                        re.compile(constraints['pattern'])
                    except re.error as e:
                        errors.append(
                            f"Constraint: Field '{field_name}' has invalid regex pattern: {str(e)}"
                        )
        
        return errors
```

---

### **STEP 2: Modify `syda/schemas.py`**

Add a validation method to `ModelConfig`:

```python
def validate_schema_config(self, schema_dict: Dict[str, Any]) -> None:
    """
    Validate schema configuration before generation.
    
    Raises:
        ValueError: If schema is invalid
    """
    if not schema_dict:
        raise ValueError("Schema cannot be empty")
    
    # Check for at least one field
    field_count = sum(1 for k, v in schema_dict.items() if not k.startswith('__'))
    if field_count == 0:
        raise ValueError("Schema must define at least one data field (not just metadata)")
    
    # Check for invalid field types
    for field_name, field_def in schema_dict.items():
        if field_name.startswith('__'):
            continue
        
        if isinstance(field_def, str):
            if field_def not in VALID_FIELD_TYPES:
                raise ValueError(f"Field '{field_name}' has invalid type '{field_def}'")
```

---

### **STEP 3: Modify `syda/generate.py`**

**Location:** In `generate_for_schemas()` method, add validation checkpoint:

```python
def generate_for_schemas(
    self,
    schemas,
    prompts=None,
    sample_sizes=None,
    output_dir=None,
    ...
):
    """Generate synthetic data for multiple schemas"""
    
    # ===== VALIDATION CHECKPOINT =====
    # Import validators
    from syda.validators import SchemaValidator, ValidationError
    
    # Initialize validator
    validator = SchemaValidator()
    
    # BEFORE loading schemas, validate raw schemas
    print("\n[INFO] Validating schemas...")
    try:
        validation_result = validator.validate_schemas(schemas)
        
        if not validation_result.is_valid:
            print(f"\n❌ SCHEMA VALIDATION FAILED ({validation_result.error_count} errors):\n")
            for schema_name, errors in validation_result.errors.items():
                if errors:
                    print(f"  {schema_name}:")
                    for error in errors:
                        print(f"    ❌ {error}")
            
            print(f"\n💡 SUGGESTIONS:")
            for suggestion in validation_result.suggestions:
                print(f"  ✓ {suggestion}")
            
            raise ValidationError(validation_result)
        else:
            print("✅ All schemas passed validation!\n")
    
    except ValidationError as e:
        raise ValueError(
            f"Schema validation failed. Cannot proceed with data generation.\n"
            f"Please fix {e.error_count} validation errors above."
        )
    
    # Continue with normal generation flow...
    if prompts is None:
        prompts = {}
    # ... rest of method
```

---

### **STEP 4: Update `syda/templates.py`**

Modify `process_template_dataframes()` to use validators:

```python
def process_template_dataframes(self, template_dataframes, output_dir=None):
    """
    Process template dataframes with pre-validation.
    """
    from syda.validators import TemplateValidator
    
    validator = TemplateValidator()
    
    for schema_name, (df, schema) in template_dataframes.items():
        # Validate template before processing
        errors = validator.validate_templates(schema_name, schema)
        
        if errors:
            print(f"\n❌ Template Validation Failed for '{schema_name}':")
            for error in errors:
                print(f"  ❌ {error}")
            raise ValueError(f"Template validation failed for {schema_name}")
        
        # Continue processing...
```

---

### **STEP 5: Integrate with Dependency Handler**

Modify `syda/dependency_handler.py` to validate FK relationships:

```python
class DependencyHandler:
    
    @staticmethod
    def validate_dependencies(
        schemas: Dict[str, Dict],
        all_dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """
        Validate all dependencies before building graph.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for circular dependencies
        try:
            # Build graph and check for cycles
            graph = DependencyHandler.build_dependency_graph(
                nodes=list(schemas.keys()),
                dependencies=all_dependencies
            )
            
            if not nx.is_directed_acyclic_graph(graph):
                cycles = list(nx.simple_cycles(graph))
                for cycle in cycles:
                    errors.append(
                        f"Circular dependency detected: {' -> '.join(cycle)} -> {cycle[0]}"
                    )
        except Exception as e:
            errors.append(f"Dependency graph error: {str(e)}")
        
        return errors
```

---

## Error Reporting Examples

### **Example 1: Foreign Key Validation Failure**

```
[INFO] Validating schemas...

❌ SCHEMA VALIDATION FAILED (3 errors):

  Order:
    ❌ FK: Field 'customer_id' references non-existent schema 'Customers'
    ❌ FK: Field 'warehouse_id' references non-existent column 'Warehouse.depot_id'
    ❌ FK: Field 'customer_id' has potentially invalid naming convention for schema 'Customer'

  Invoice:
    ❌ FK: Field 'order_fk' has potentially invalid naming convention for schema 'Order'

💡 SUGGESTIONS:
  ✓ Verify schema names match exactly: 'Customers' should be 'Customer'?
  ✓ Check target column names: Did you mean 'warehouse.id' instead of 'warehouse.depot_id'?
  ✓ Use naming convention 'order_id' instead of 'order_fk' for better inference
  ✓ Consider using explicit foreign key definitions instead of naming convention inference
```

### **Example 2: Template Placeholder Validation Failure**

```
[INFO] Validating schemas...

❌ SCHEMA VALIDATION FAILED (2 errors):

  Invoice:
    ❌ Template: Placeholder '{{ customer_phone }}' is not defined in schema
    ❌ Template: Placeholder '{{ invoice_date }}' is not defined in schema
    ❌ Template: Missing '__input_file_type__' metadata

💡 SUGGESTIONS:
  ✓ Add missing fields to schema: 'customer_phone', 'invoice_date'
  ✓ Add '__input_file_type__': 'html' to schema metadata
  ✓ Template file at '/templates/invoice.html' contains 3 placeholders, schema only defines 1
```

### **Example 3: Constraint Validation Failure**

```
❌ SCHEMA VALIDATION FAILED (1 error):

  Product:
    ❌ Constraint: Field 'price' has min > max (1000 > 100)
    ❌ Constraint: Field 'sku' has invalid regex pattern: 'invalid[' (unbalanced bracket)

💡 SUGGESTIONS:
  ✓ Fix constraint range for 'price': min should be ≤ max
  ✓ Review regex pattern for 'sku': did you mean 'invalid\\['?
```

---

## Backward Compatibility

### **Strategy:**
1. **Validation is NOT breaking by default** - Add `validate_before_generation: bool = True` parameter
2. Existing code continues to work but logs warnings
3. New strict mode available via `strict_validation=True`

### **Example:**

```python
# Default behavior (validates, but logs errors instead of raising)
generator.generate_for_schemas(
    schemas=schemas,
    validate_before_generation=True,  # Logs errors
    strict_validation=False  # Default: continue on warnings
)

# Strict mode (raises exception on any validation error)
generator.generate_for_schemas(
    schemas=schemas,
    strict_validation=True  # Raises ValidationError
)

# Disable validation (for backward compatibility)
generator.generate_for_schemas(
    schemas=schemas,
    validate_before_generation=False  # Skip validation entirely
)
```

---

## Testing Strategy

### **Unit Tests** (`tests/test_validators.py`)

```python
# Test foreign key validation
def test_foreign_key_validator_missing_table():
    """Should detect missing target table"""
    
def test_foreign_key_validator_missing_column():
    """Should detect missing target column"""
    
def test_foreign_key_validator_circular_dependency():
    """Should detect circular FK references"""

# Test template validation
def test_template_validator_missing_placeholder():
    """Should detect undefined placeholders"""
    
def test_template_validator_missing_file():
    """Should detect non-existent template files"""
    
def test_template_validator_missing_metadata():
    """Should detect missing template metadata"""

# Test constraint validation
def test_constraint_validator_invalid_range():
    """Should detect min > max"""
    
def test_constraint_validator_invalid_regex():
    """Should detect invalid regex patterns"""
```

### **Integration Tests**

```python
def test_generate_with_invalid_schema():
    """Should raise ValidationError before generation"""
    
def test_generate_with_valid_schema():
    """Should pass validation and generate successfully"""
```

---

## Implementation Roadmap

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| **1** | Create `syda/validators.py` with core validator classes | 2-3 hrs | CRITICAL |
| **2** | Implement `ForeignKeyValidator` | 1-2 hrs | CRITICAL |
| **3** | Implement `TemplateValidator` | 1-2 hrs | CRITICAL |
| **4** | Implement `ConstraintValidator` | 1 hr | HIGH |
| **5** | Integrate validators into `generate_for_schemas()` | 1 hr | HIGH |
| **6** | Add validation to `templates.py` | 30 min | HIGH |
| **7** | Add `validate_before_generation` parameter | 30 min | MEDIUM |
| **8** | Create unit tests (`test_validators.py`) | 2-3 hrs | HIGH |
| **9** | Create integration tests | 1-2 hrs | HIGH |
| **10** | Update documentation with examples | 1 hr | MEDIUM |
| **TOTAL** | | **11-15 hrs** | |

---

## Key Benefits

✅ **Fail Fast**: Catch schema errors before expensive AI calls
✅ **Clear Error Messages**: Users understand exactly what's wrong
✅ **Backward Compatible**: Existing code works with warnings
✅ **Actionable Suggestions**: Hints for fixing common issues
✅ **Data Integrity**: Prevents corrupted datasets
✅ **Development Efficiency**: Faster debugging for users
✅ **Production Ready**: Suitable for automated pipelines

---

## Configuration Example

```python
from syda import SyntheticDataGenerator
from syda.validators import SchemaValidator

# Create generator
generator = SyntheticDataGenerator()

# Define schemas (with potential issues)
schemas = {
    'customers': {
        'id': 'integer',
        'name': 'text',
        'email': 'email'
    },
    'orders': {
        '__foreign_keys__': {
            'cust_id': ('customer', 'id')  # ❌ Wrong table name
        },
        'id': 'integer',
        'cust_id': 'foreign_key',
        'total': 'number'
    }
}

# Generate with validation
try:
    results = generator.generate_for_schemas(
        schemas=schemas,
        sample_sizes={'customers': 100, 'orders': 500},
        strict_validation=True  # Fail on first error
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Fix and retry
```

