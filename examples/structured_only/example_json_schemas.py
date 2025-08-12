#!/usr/bin/env python
"""
Example of using the SyntheticDataGenerator to handle multiple related JSON schema files
with embedded foreign key relationships.

This example demonstrates:
1. Loading schemas exclusively from JSON files
2. Foreign key relationships defined within the schema files themselves
3. Automatic dependency resolution between schemas
4. Custom generators for specific fields
"""

import sys
import os
import random
from dotenv import load_dotenv
import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the synthetic data generator
from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig


def main():
    """Demonstrate automatic generation of related data using JSON schema files with embedded foreign keys."""
    
    # Create a generator instance with appropriate max_tokens setting
    model_config = ModelConfig(
        provider="anthropic",
        #model_name="gpt-4",  # Default model
        model_name="claude-3-5-haiku-20241022",
        temperature=0.7,
        max_tokens=8192,  # Using higher max_tokens value for more complete responses
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Define output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "output", 
        "example_json_schemas", 
        "blog_data"
    )
    
    # Define paths to schema files
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_files/json")
    
    # Dictionary mapping schema names to their JSON file paths
    schemas = {
        "User": os.path.join(schema_dir, "user.json"),
        "Post": os.path.join(schema_dir, "post.json"),
        "Comment": os.path.join(schema_dir, "comment.json")
    }
    
    # Define custom prompts for each schema (optional)
    prompts = {
        "User": """
        Generate diverse users for a blog platform.
        Include regular users and some administrators.
        Create a range of join dates over the past 5 years.
        """,
        
        "Post": """
        Generate blog posts with diverse titles and categories.
        Include topics like technology, travel, food, health, and business.
        Create realistic content snippets and publication dates.
        """,
        
        "Comment": """
        Generate diverse comments on blog posts.
        Include short comments, longer thoughtful responses, and questions.
        Create a mix of approved and unapproved comments.
        """
    }
    
    # Define sample sizes for each schema (optional)
    sample_sizes = {
        "User": 15,      # Base entities
        "Post": 30,      # Posts by the users (~2 per user)
        "Comment": 60,   # Comments on posts (~2 per post)
    }
    
    # Define custom generators for specific schema columns
    #
    # NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
    # based on the schema definitions. Custom generators give you precise control for fields where
    # you need specific distributions or formatting that might be challenging for the AI.
    #
    # Below we include just a few key custom generators as examples:
    #
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
    
    print("\n🔄 Generating related data for Blog system...")
    print("  The system will automatically extract foreign key relationships from schema files")
    print("  and determine the correct generation order\n")
    
    # Generate data using schemas with embedded foreign keys
    # Note: We don't need to explicitly provide the foreign_keys parameter
    results = generator.generate_for_schemas(
        schemas=schemas,
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=output_dir,
        custom_generators=custom_generators
    )
    
    # Print summary
    print("\n✅ Data generation complete!")
    for schema_name, df in results.items():
        print(f"  - {schema_name}: {len(df)} records")
    
    print(f"\nData files saved to directory: {output_dir}/")
    
    # Show samples of data with custom generators
    print("\n📊 Sample data statistics:")
    
    print("\nUser admin distribution:")
    admin_counts = results["User"]["is_admin"].value_counts().to_dict()
    print(f"  - Admins: {admin_counts.get(True, 0)} ({admin_counts.get(True, 0)/len(results['User'])*100:.1f}%)")
    print(f"  - Regular users: {admin_counts.get(False, 0)} ({admin_counts.get(False, 0)/len(results['User'])*100:.1f}%)")
    
    print("\nPost categories distribution:")
    category_counts = results["Post"]["category"].value_counts().to_dict()
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {category}: {count} posts")
    
    print("\nPost status distribution:")
    status_counts = results["Post"]["status"].value_counts().to_dict()
    for status, count in status_counts.items():
        percentage = (count / len(results["Post"])) * 100
        print(f"  - {status}: {count} posts ({percentage:.1f}%)")
    
    print("\n🔍 Verifying referential integrity:")
    
    # Check Posts → Users
    post_author_ids = set(results["Post"]["author_id"].tolist())
    valid_user_ids = set(results["User"]["id"].tolist())
    if post_author_ids.issubset(valid_user_ids):
        print("  ✅ All Post.author_id values reference valid Users")
    else:
        print("  ❌ Invalid Post.author_id references detected")
    
    # Check Comments → Posts
    comment_post_ids = set(results["Comment"]["post_id"].tolist())
    valid_post_ids = set(results["Post"]["id"].tolist())
    if comment_post_ids.issubset(valid_post_ids):
        print("  ✅ All Comment.post_id values reference valid Posts")
    else:
        print("  ❌ Invalid Comment.post_id references detected")
    
    # Check Comments → Users
    comment_user_ids = set(results["Comment"]["user_id"].tolist())
    if comment_user_ids.issubset(valid_user_ids):
        print("  ✅ All Comment.user_id values reference valid Users")
    else:
        print("  ❌ Invalid Comment.user_id references detected")
        
    # Display comment hierarchy information
    top_level_comments = results["Comment"][results["Comment"]["parent_comment_id"] == 0]
    threaded_comments = results["Comment"][results["Comment"]["parent_comment_id"] > 0]
    print(f"  ℹ️ Comment hierarchy: {len(top_level_comments)} top-level comments, {len(threaded_comments)} threaded replies")


if __name__ == "__main__":
    main()
