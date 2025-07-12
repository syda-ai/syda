## Installation

Install the package using pip:

```bash
pip install syda
```
## Prepare environment variables

Before running the example, you need to prepare your environment variables with valid OpenAI or Anthropic (Claude) API keys. You can do this in one of two ways:

1. **Create a `.env` file**:

    Create a `.env` file in the root of this project with the following content:

    ```
    OPENAI_API_KEY=your_openai_key
    ANTHROPIC_API_KEY=your_anthropic_key
    ```

2. **Set environment variables directly**:

    You can also set the environment variables directly in your code:

    ```bash
    export OPENAI_API_KEY=your_openai_key
    export ANTHROPIC_API_KEY=your_anthropic_key
    ```
## Example
```python
from syda.structured import SyntheticDataGenerator
from syda.schemas import ModelConfig
import os

# Create a model config instance with appropriate max_tokens setting
model_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    temperature=0.7,
    max_tokens=8192,  # Using higher max_tokens value for more complete responses
)

# Create a syda generator instance and pass the model config
generator = SyntheticDataGenerator(model_config=model_config)

# Define output directory
output_dir = "synthetic_output/ecommerce"

# Define schema dictionaries for an e-commerce system with descriptions
schemas = {
    # Customer schema with table and column descriptions
    'Customer': {
        # Define schema with additional metadata
        '__table_description__': 'Registered users of the e-commerce platform who can place orders',
        'id': {'type': 'number', 'description': 'Unique identifier for the customer'},
        'name': {'type': 'text', 'description': 'Full name of the customer'},
        'email': {'type': 'email', 'description': 'Customer email address used for communication'},
        'signup_date': {'type': 'date', 'description': 'Date when the customer registered'},
        'loyalty_tier': {'type': 'text', 'description': 'Customer loyalty program level (Bronze, Silver, Gold, Platinum)'}
    },
    
    # Product schema with table and column descriptions
    'Product': {
        '__table_description__': 'Products available for purchase in the e-commerce store',
        'id': {'type': 'number', 'description': 'Unique identifier for the product'},
        'name': {'type': 'text', 'description': 'Name of the product as displayed to customers'},
        'category': {'type': 'text', 'description': 'Product category for classification and filtering'},
        'price': {'type': 'number', 'description': 'Current price of the product in USD'},
        'description': {'type': 'text', 'description': 'Detailed description of the product features and benefits'},
        'in_stock': {'type': 'boolean', 'description': 'Whether the product is currently available for purchase'}
    },
    
    # Order schema with table and column descriptions
    'Order': {
        # Define schema with additional metadata
        '__table_description__': 'Customer orders for products, including order status and total amount',
        '__foreign_keys__': {
            'customer_id': ['Customer', 'id']  # Order.customer_id references Customer.id
        },
        
        # Define columns
        'id': {'type': 'number', 'description': 'Unique order identifier', 'primary_key': True},
        'customer_id': {'type': 'foreign_key', 'description': 'Reference to the customer who placed the order'},
        'order_date': {'type': 'date', 'description': 'Date when the order was placed'},
        'status': {'type': 'text', 'description': 'Current status of the order (Pending, Processing, Shipped, Delivered, Cancelled)'},
        'total_amount': {'type': 'number', 'description': 'Total amount of the order in USD'},
        'shipping_address': {'type': 'text', 'description': 'Address where the order should be delivered'}
    },
    
    # OrderItem schema with table and column descriptions
    'OrderItem': {
        '__table_description__': 'Individual line items within an order, representing specific products',
        '__foreign_keys__': {
            'order_id': ['Order', 'id'],       # OrderItem.order_id references Order.id
            'product_id': ['Product', 'id']    # OrderItem.product_id references Product.id
        },
        
        # Define columns
        'id': {'type': 'number', 'description': 'Unique identifier for the order item'},
        'order_id': {'type': 'foreign_key', 'description': 'Reference to the parent order'},
        'product_id': {'type': 'foreign_key', 'description': 'Reference to the product being ordered'},
        'quantity': {'type': 'number', 'description': 'Number of units of the product ordered'},
        'unit_price': {'type': 'number', 'description': 'Price per unit at the time of order, may differ from current product price'}
    }
}

# Define foreign key relationships
foreign_keys = {
    'Order': {
        'customer_id': ('Customer', 'id')  # Order.customer_id references Customer.id
    },
    'OrderItem': {
        'order_id': ('Order', 'id'),       # OrderItem.order_id references Order.id
        'product_id': ('Product', 'id')    # OrderItem.product_id references Product.id
    }
}

# Define custom prompts for each schema (optional)
prompts = {
    "Customer": """
    Generate diverse customers for an e-commerce platform.
    Include various loyalty tiers (Bronze, Silver, Gold, Platinum)
    and realistic signup dates within the last 3 years.
    """,
    
    "Product": """
    Generate diverse products for an e-commerce store.
    Include various categories (Electronics, Clothing, Home, Books, etc.)
    with realistic prices and descriptions.
    """,
    
    "Order": """
    Generate realistic orders with appropriate dates and statuses
    (Pending, Processing, Shipped, Delivered, Cancelled).
    Total amounts should reflect typical e-commerce purchases.
    """
}

# Define sample sizes for each schema (optional)
sample_sizes = {
    "Customer": 10,       # Base entities
    "Product": 15,        # Product catalog
    "Order": 25,          # ~2-3 orders per customer
    "OrderItem": 50,      # ~2 items per order
}

print("\n🔄 Generating related data for E-commerce system...")
print("  The system will automatically determine the right generation order")
print("  and set up foreign key relationships\n")

# Generate data using the unified generate_for_schemas method
results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes=sample_sizes,
    output_dir=output_dir
)
```