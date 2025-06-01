#!/usr/bin/env python
"""
Example of using the enhanced SyntheticDataGenerator to automatically handle
multiple related SQLAlchemy models with foreign key relationships.

This example demonstrates:
1. Defining a set of related SQLAlchemy models with foreign key relationships
2. Using generate_for_sqlalchemy_models to automatically handle:
   - Dependency resolution between models
   - Generation order (parents before children)
   - Foreign key constraints
   - Column-specific generators
3. All without manually managing the process
"""

import sys
import os
import random
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Date, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship
import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the synthetic data generator
from syda.generate import SyntheticDataGenerator

# Create a Base for our models
Base = declarative_base()

##############################################################################
# Define a comprehensive set of related models for a CRM system
##############################################################################

class Customer(Base):
    """Organization or individual client in the CRM system.
    Represents a business entity that can place orders and have contacts.
    """
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True, 
                 comment="Customer organization name")
    industry = Column(String(50), comment="Customer's primary industry")
    website = Column(String(100), comment="Customer's website URL")
    status = Column(String(20), comment="Active, Inactive, Prospect")
    created_at = Column(Date, default=datetime.date.today,
                       comment="Date when customer was added to CRM")
    
    # Relationships
    contacts = relationship("Contact", back_populates="customer")
    orders = relationship("Order", back_populates="customer")


class Contact(Base):
    """Individual person associated with a customer organization.
    Contacts are the individuals we communicate with at the customer.
    """
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False,
                        comment="Customer this contact belongs to")
    first_name = Column(String(50), nullable=False, 
                       comment="Contact's first name")
    last_name = Column(String(50), nullable=False, 
                      comment="Contact's last name")
    email = Column(String(100), nullable=False, unique=True, 
                  comment="Contact's email address")
    phone = Column(String(20), comment="Contact's phone number")
    position = Column(String(100), comment="Job title or position")
    is_primary = Column(Boolean, default=False, 
                       comment="Whether this is the primary contact")
    
    # Relationships
    customer = relationship("Customer", back_populates="contacts")


class Product(Base):
    """Product or service offered by the company.
    Products can be ordered by customers.
    """
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True,
                 comment="Product name")
    category = Column(String(50), comment="Product category")
    price = Column(Float, nullable=False, comment="Product price in USD")
    description = Column(Text, comment="Detailed product description")
    
    # Relationships
    order_items = relationship("OrderItem", back_populates="product")


class Order(Base):
    """Customer order for products or services.
    Orders contain order items for specific products.
    """
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False,
                        comment="Customer who placed the order")
    order_date = Column(Date, nullable=False, 
                       comment="Date when order was placed")
    status = Column(String(20), comment="New, Processing, Shipped, Delivered, Cancelled")
    total_amount = Column(Float, comment="Total order amount in USD")
    
    # Relationships
    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    """Individual item within an order.
    Each order item represents a specific product in a specific quantity.
    """
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False,
                     comment="Order this item belongs to")
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False,
                       comment="Product being ordered")
    quantity = Column(Integer, nullable=False, 
                     comment="Quantity of product ordered")
    unit_price = Column(Float, nullable=False, 
                       comment="Price per unit at time of order")
    
    # Relationships
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


def main():
    """Demonstrate automatic generation of related data with proper foreign key handling."""
    
    # Create a generator instance with appropriate max_tokens setting and logging enabled
    from syda.schemas import ModelConfig    
    
    model_config = ModelConfig(
        provider="anthropic",
        #model_name="claude-3-opus-20240229",  
        model_name="claude-3-5-haiku-20241022",
        temperature=0.7,
        max_tokens=4000,  # Using higher max_tokens value for more complete responses
    )
    generator = SyntheticDataGenerator(model_config=model_config)
    
    # Define output directory
    output_dir = "crm_data"
    
    # Define custom prompts for each model (optional)
    prompts = {
        "customers": """
        Generate diverse customer organizations for a B2B SaaS company.
        Include a mix of industries like technology, healthcare, finance, etc.
        """,
        
        "products": """
        Generate products for a cloud software company.
        Products should include various software services, support packages, and consulting.
        """,
        
        "orders": """
        Generate realistic orders with appropriate dates and statuses.
        """,
    }
    
    # Define sample sizes for each model (optional)
    sample_sizes = {
        "customers": 10,        # Base entities
        "contacts": 25,         # ~2-3 contacts per customer
        "products": 15,         # Products catalog
        "orders": 30,           # ~3 orders per customer
        "order_items": 60,       # ~2 items per order
    }
    
    # Define custom generators for specific model columns
    #
    # NOTE: Custom generators are OPTIONAL. The AI will generate reasonable values for most fields
    # based on column names, SQLAlchemy comments, and field types. Custom generators give you precise 
    # control for fields where you need specific distributions or formatting.
    #
    # This example shows a balanced approach with just a few strategic custom generators:
    #
    custom_generators = {
        "customers": {
            # Ensure a specific distribution of customer statuses for business reporting
            "status": lambda row, col: random.choice(["Active", "Inactive", "Prospect"]),
        },
        "products": {
            # Control price ranges more precisely than the AI might
            "price": lambda row, col: round(random.uniform(50, 5000), 2),
            # Ensure product categories match your specific business domains
            "category": lambda row, col: random.choice([
                "Cloud Infrastructure", "Business Intelligence", "Security Services",
                "Data Analytics", "Custom Development", "Support Package", "API Services"
            ])
        },
        "order_items": {
            # Example of a simple numeric distribution
            "quantity": lambda row, col: random.randint(1, 10),
        },
    }
    
    print("\n🔄 Generating related data for CRM system...")
    print("  The system will automatically determine the right generation order")
    print("  and set up foreign key relationships")
    print("  with custom generators for specific columns\n")
    
    # Generate data for all SQLAlchemy models with automatic dependency resolution
    results = generator.generate_for_sqlalchemy_models(
        sqlalchemy_models=[Customer, Contact, Product, Order, OrderItem],
        prompts=prompts,
        sample_sizes=sample_sizes,
        output_dir=output_dir,
        custom_generators=custom_generators
    )
    
    # Print summary
    print("\n✅ Data generation complete!")
    for model_name, df in results.items():
        print(f"  - {model_name}: {len(df)} records")
    
    print(f"\nData files saved to directory: {output_dir}/")
    
    # Show samples of data with custom generators
    print("\n📊 Sample data with custom generators:")
    
    print("\nCustomer statuses (custom generator):")
    status_counts = results["customers"]["status"].value_counts().to_dict()
    for status, count in status_counts.items():
        print(f"  - {status}: {count} customers")
    
    print("\nProduct categories (custom generator):")
    category_counts = results["products"]["category"].value_counts().to_dict()
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {category}: {count} products")
    
    print("\nProduct prices (custom generator):")
    price_min = results["products"]["price"].min()
    price_max = results["products"]["price"].max()
    price_avg = results["products"]["price"].mean()
    print(f"  - Price range: ${price_min:.2f} to ${price_max:.2f}")
    print(f"  - Average price: ${price_avg:.2f}")
    
    # Verify referential integrity
    print("\n🔍 Verifying referential integrity:")
    
    # Check Contacts → Customers
    contact_customer_ids = set(results["contacts"]["customer_id"].tolist())
    valid_customer_ids = set(results["customers"]["id"].tolist())
    if contact_customer_ids.issubset(valid_customer_ids):
        print("  ✅ All Contact.customer_id values reference valid Customers")
    else:
        print("  ❌ Invalid Contact.customer_id references detected")
    
    # Check Orders → Customers
    order_customer_ids = set(results["orders"]["customer_id"].tolist())
    if order_customer_ids.issubset(valid_customer_ids):
        print("  ✅ All Order.customer_id values reference valid Customers")
    else:
        print("  ❌ Invalid Order.customer_id references detected")
    
    # Check OrderItems → Orders
    order_item_order_ids = set(results["order_items"]["order_id"].tolist())
    valid_order_ids = set(results["orders"]["id"].tolist())
    if order_item_order_ids.issubset(valid_order_ids):
        print("  ✅ All OrderItem.order_id values reference valid Orders")
    else:
        print("  ❌ Invalid OrderItem.order_id references detected")
    
    # Check OrderItems → Products
    order_item_product_ids = set(results["order_items"]["product_id"].tolist())
    valid_product_ids = set(results["products"]["id"].tolist())
    if order_item_product_ids.issubset(valid_product_ids):
        print("  ✅ All OrderItem.product_id values reference valid Products")
    else:
        print("  ❌ Invalid OrderItem.product_id references detected")


if __name__ == "__main__":
    main()
