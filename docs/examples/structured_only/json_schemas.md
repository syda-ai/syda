# JSON Schema Examples

> Source code: [examples/structured_only/example_json_schemas.py](https://github.com/syda-ai/syda/blob/main/examples/structured_only/example_json_schemas.py)

This example demonstrates how to define and use JSON-based schemas for synthetic data generation with SYDA.

## Overview

JSON schemas provide a structured way to define your data models in external files. This approach is ideal for complex data structures and when you want to maintain separation between schema definitions and code logic.

## Schema Definition

JSON schemas use a structured format where:
- Each JSON file defines one schema/table
- Field definitions include types and descriptions
- Special keys like `__table_description__` provide metadata
- Constraints and foreign keys are explicitly defined

Here are examples of JSON schema files for a blog system:

### User Schema (user.json)

```json
{
  "__table_description__": "User accounts for the blog system",
  "id": {
    "type": "number",
    "description": "Unique identifier for the user",
    "constraints": {
      "primary_key": true
    }
  },
  "username": {
    "type": "text",
    "description": "User's login name",
    "constraints": {
      "unique": true,
      "min_length": 3,
      "max_length": 50
    }
  },
  "email": {
    "type": "email",
    "description": "User's email address",
    "constraints": {
      "unique": true,
      "max_length": 150
    }
  },
  "full_name": {
    "type": "text",
    "description": "User's full name",
    "constraints": {
      "max_length": 100
    }
  },
  "join_date": {
    "type": "date",
    "description": "Date when the user registered"
  },
  "bio": {
    "type": "text",
    "description": "User's biographical information"
  },
  "is_admin": {
    "type": "boolean",
    "description": "Whether the user has administrator privileges"
  }
}
```

### Post Schema (post.json)

```json
{
  "__table_description__": "Blog posts created by users",
  "id": {
    "type": "number",
    "description": "Unique identifier for the post",
    "constraints": {
      "primary_key": true
    }
  },
  "author_id": {
    "type": "foreign_key",
    "description": "Reference to the user who created the post",
    "references": {
      "schema": "User",
      "field": "id"
    }
  },
  "title": {
    "type": "text",
    "description": "Title of the blog post",
    "constraints": {
      "unique": true,
      "max_length": 200
    }
  },
  "content": {
    "type": "text",
    "description": "Full content of the blog post",
    "constraints": {
      "max_length": 50000
    }
  },
  "publish_date": {
    "type": "date",
    "description": "Date when the post was published"
  },
  "category": {
    "type": "text",
    "description": "Category of the blog post",
    "constraints": {
      "max_length": 50
    }
  },
  "tags": {
    "type": "text",
    "description": "Comma-separated list of tags for the post",
    "constraints": {
      "max_length": 500
    }
  }
}
```

### Comment Schema (comment.json)

```json
{
  "__table_description__": "Comments on blog posts by users",
  "__foreign_keys__": {
    "post_id": ["Post", "id"],
    "user_id": ["User", "id"]
  },
  "id": {
    "type": "number",
    "description": "Unique identifier for the comment",
    "constraints": {
      "primary_key": true
    }
  },
  "post_id": {
    "type": "foreign_key",
    "description": "Reference to the post being commented on",
    "references": {
      "schema": "Post",
      "field": "id"
    }
  },
  "user_id": {
    "type": "foreign_key",
    "description": "Reference to the user who wrote the comment",
    "references": {
      "schema": "User",
      "field": "id"
    }
  },
  "content": {
    "type": "text",
    "description": "Content of the comment",
    "constraints": {
      "max_length": 1000
    }
  },
  "created_at": {
    "type": "date",
    "description": "Date and time when the comment was created"
  },
  "is_approved": {
    "type": "boolean",
    "description": "Whether the comment has been approved by moderators"
  },
  "parent_comment_id": {
    "type": "number",
    "description": "Reference to the parent comment (for threaded comments), indicate 0 if it is a top-level comment or if it is a parent comment on the post"
  }
}
```

## Schema Directory Structure

For a blog system, you might structure your JSON schemas as follows:

```
schema_files/json/
├── user.json
├── post.json
└── comment.json
```

## Foreign Key Handling

JSON schemas support three methods for defining foreign key relationships:

### 1. Using Field-Level References (Recommended for JSON)

```json
{
  "user_id": {
    "type": "foreign_key",
    "description": "Reference to the author of the post",
    "references": {
      "schema": "User",
      "field": "id"
    }
  }
}
```

### 2. Using the `__foreign_keys__` Special Section

```json
{
  "__foreign_keys__": {
    "user_id": ["User", "id"]
  }
}
```

## Code Example

Here's how to use JSON-based schemas with the SyntheticDataGenerator:

```python
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig
import os
import random
import datetime

# Create a generator instance
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192
)
generator = SyntheticDataGenerator(model_config=model_config)

# Define paths to schema files
schema_dir = "schema_files/json"
schemas = {
    "User": os.path.join(schema_dir, "user.json"),
    "Post": os.path.join(schema_dir, "post.json"),
    "Comment": os.path.join(schema_dir, "comment.json")
}

# Define custom prompts
prompts = {
    "User": "Generate diverse users for a blog platform...",
    "Post": "Generate blog posts with diverse titles and categories...",
    "Comment": "Generate diverse comments on blog posts..."
}

# Define sample sizes
sample_sizes = {
    "User": 15,      # Base entities
    "Post": 30,      # Posts by the users (~2 per user)
    "Comment": 60,   # Comments on posts (~2 per post)
}

# Define custom generators for specific schema fields
# NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
# based on the schema definitions. Custom generators give you precise control for fields where
# you need specific distributions or formatting that might be challenging for the AI.
custom_generators = {
    "User": {
        # Generate a specific distribution of admin users
        "is_admin": lambda row, col: random.choices([True, False], weights=[0.2, 0.8])[0],
    },
    "Post": {
        # Control publication dates to follow a specific timeline
        "publish_date": lambda row, col: (datetime.datetime.now() - 
            datetime.timedelta(days=random.randint(0, 365*2))).strftime("%Y-%m-%d"),
        # Ensure post statuses follow business rules with specific ratios
        "status": lambda row, col: random.choices(
            ["published", "draft", "archived"],
            weights=[0.7, 0.2, 0.1]
        )[0]
    }
}

# Generate data
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir="output/example_json_schemas/blog_data",
    custom_generators=custom_generators
)
```

## Key Features

1. **Structured Format**: JSON provides a well-defined structure for complex schemas
2. **External Definition**: Maintain schema definitions separate from code
3. **Rich Constraints**: Define constraints like uniqueness, min/max values
4. **Automatic Generation Order**: SYDA handles dependency resolution between tables

## Best Practices

1. **Include Constraints**: Use constraints to guide the AI generation process
2. **Explicit Foreign Keys**: Always define foreign key relationships clearly
3. **Comprehensive Descriptions**: Add detailed descriptions to help the AI generate appropriate data
4. **Schema Validation**: Validate JSON schemas to ensure correctness
5. **Self-Referential Handling**: When designing hierarchical data, use the appropriate foreign key pattern

## Sample Outputs

You can view sample outputs generated using these JSON schemas here:

> [Example JSON Schema Outputs](https://github.com/syda-ai/syda/tree/main/examples/structured_only/output/example_json_schemas/blog_data)
