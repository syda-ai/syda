#!/usr/bin/env python
"""
Example of using the enhanced SyntheticDataGenerator to automatically handle
multiple related schemas stored in both JSON and YAML files with foreign key relationships.

This example demonstrates:
1. Loading schemas from JSON and YAML files
2. Using generate_for_schemas to automatically handle:
   - Dependency resolution between schemas
   - Generation order (parents before children)
   - Foreign key constraints
   - Column-specific generators
3. All without manually managing the process
"""

import sys
import os
import random
from dotenv import load_dotenv
import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the synthetic data generator
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig


def main():
    """Demonstrate automatic generation of related data using JSON and YAML schema files."""
    
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
    output_dir = "publishing_data"
    
    # Define paths to schema files
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_files")
    
    # Dictionary mapping schema names to their file paths (JSON and YAML)
    schemas = {
        "Author": os.path.join(schema_dir, "author.json"),
        "Book": os.path.join(schema_dir, "book.yml"),       # YAML format
        "Review": os.path.join(schema_dir, "review.json")
    }
    
    # Define foreign key relationships between schemas
    foreign_keys = {
        "Book": {
            "author_id": ("Author", "id")  # Book.author_id references Author.id
        },
        "Review": {
            "book_id": ("Book", "id")      # Review.book_id references Book.id
        }
    }
    
    # Define custom prompts for each schema (optional)
    prompts = {
        "Author": """
        Generate diverse authors for a publishing system.
        Include a mix of fiction and non-fiction writers across different genres,
        with realistic biographical information and birth dates.
        """,
        
        "Book": """
        Generate books with diverse titles, genres, and publication dates.
        Include fiction (mystery, sci-fi, romance, fantasy) and non-fiction
        (biography, history, science, self-help) with realistic descriptions.
        """,
        
        "Review": """
        Generate realistic book reviews with varied ratings (1-5 stars) and
        diverse reviewer sentiments. Include both critical and positive reviews.
        """
    }
    
    # Define sample sizes for each schema (optional)
    sample_sizes = {
        "Author": 12,      # Base entities
        "Book": 25,        # Books by the authors (~2 per author)
        "Review": 40,      # Reviews for the books (~1-2 per book)
    }
    
    # Define custom generators for specific schema columns
    custom_generators = {
        "Author": {
            # Generate birth dates between 1940 and 1990
            "birth_date": lambda row, col: (datetime.datetime(1940, 1, 1) + 
                datetime.timedelta(days=random.randint(0, 365*50))).strftime("%Y-%m-%d"),
        },
        "Book": {
            # Generate prices between $5 and $40 with appropriate precision
            "price": lambda row, col: round(random.uniform(5, 40), 2),
            # Make book genres more specific
            "genre": lambda row, col: random.choice([
                "Mystery", "Science Fiction", "Fantasy", "Romance", "Literary Fiction",
                "Biography", "History", "Science", "Self-Help", "Thriller", "Horror",
                "Memoir", "Poetry", "Young Adult", "Children's", "Cookbook"
            ]),
            # Generate page counts between 100 and 800
            "page_count": lambda row, col: random.randint(100, 800)
        },
        "Review": {
            # Generate realistic ratings with a bias toward positive reviews
            "rating": lambda row, col: random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.05, 0.1, 0.2, 0.35, 0.3]  # More likely to be 4-5 stars
            )[0],
            # Randomly set verified purchase flag
            "verified_purchase": lambda row, col: random.choices(
                [True, False],
                weights=[0.8, 0.2]  # 80% are verified purchases
            )[0]
        },
    }
    
    print("\n🔄 Generating related data for Publishing system...")
    print("  The system will automatically determine the right generation order")
    print("  and set up foreign key relationships between JSON and YAML schema files")
    print("  with custom generators for specific columns\n")
    
    # Generate data using the unified generate_for_schemas method
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
    print("\n📊 Sample data with custom generators:")
    
    print("\nBook genres (custom generator):")
    genre_counts = results["Book"]["genre"].value_counts().to_dict()
    for genre, count in list(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))[:5]:
        print(f"  - {genre}: {count} books")
    
    print("\nBook prices (custom generator):")
    prices = results["Book"]["price"].tolist()
    print(f"  - Price range: ${min(prices):.2f} to ${max(prices):.2f}")
    print(f"  - Average price: ${sum(prices)/len(prices):.2f}")
    
    print("\nReview ratings distribution (custom generator):")
    rating_counts = results["Review"]["rating"].value_counts().to_dict()
    for rating in sorted(rating_counts.keys()):
        count = rating_counts[rating]
        percentage = (count / len(results["Review"])) * 100
        print(f"  - {rating} stars: {count} reviews ({percentage:.1f}%)")
    
    print("\n🔍 Verifying referential integrity:")
    
    # Check Books → Authors
    book_author_ids = set(results["Book"]["author_id"].tolist())
    valid_author_ids = set(results["Author"]["id"].tolist())
    if book_author_ids.issubset(valid_author_ids):
        print("  ✅ All Book.author_id values reference valid Authors")
    else:
        print("  ❌ Invalid Book.author_id references detected")
    
    # Check Reviews → Books
    review_book_ids = set(results["Review"]["book_id"].tolist())
    valid_book_ids = set(results["Book"]["id"].tolist())
    if review_book_ids.issubset(valid_book_ids):
        print("  ✅ All Review.book_id values reference valid Books")
    else:
        print("  ❌ Invalid Review.book_id references detected")


if __name__ == "__main__":
    main()
