"""
CRM Example models using SQLAlchemy with template support.
"""
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from syda.templates import SydaTemplate
import os

Base = declarative_base()

# Structured data models
class Customer(Base):
    """Customer organization in the CRM system."""
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    industry = Column(String(50))
    annual_revenue = Column(Float)
    employees = Column(Integer)
    website = Column(String(100))
    address = Column(String(200))
    city = Column(String(50))
    state = Column(String(2))
    zip_code = Column(String(10))
    status = Column(String(20))  # Active, Inactive, Prospect
    
    # Relationships
    contacts = relationship("Contact", back_populates="customer")
    opportunities = relationship("Opportunity", back_populates="customer")


class Contact(Base):
    """Individual person associated with a customer."""
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    phone = Column(String(20))
    position = Column(String(100))
    is_primary = Column(Boolean)
    
    # Relationships
    customer = relationship("Customer", back_populates="contacts")


class Opportunity(Base):
    """Sales opportunity with a customer."""
    __tablename__ = 'opportunities'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    stage = Column(String(20), nullable=False)  # Lead, Qualification, Proposal, Negotiation, Closed Won, Closed Lost
    probability = Column(Float)
    expected_close_date = Column(Date)
    description = Column(Text)
    
    # Relationships
    customer = relationship("Customer", back_populates="opportunities")

templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Template models that depend on structured data
class ProposalDocument(SydaTemplate):
    """Sales proposal document for an opportunity."""
    # Special metadata attributes
    __template__ = True
    __depends_on__ = ['Opportunity']
    
    # Template configuration as regular fields (these become columns in the generated data)
    __template_source__ = os.path.join(templates_dir, 'proposal.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    # Fields needed for the template
    opportunity_id = None  # This will be filled in at runtime
    created_date = None
    title = None
    subtitle = None
    prepared_by = None
    customer_name = None
    customer_address = None
    opportunity_name = None
    opportunity_value = None
    opportunity_description = None
    proposed_solutions = None
    implementation_timeline = None
    pricing_details = None
    terms_and_conditions = None


class ContractDocument(SydaTemplate):
    """Contract document for a won opportunity."""
    # Special metadata attributes
    __template__ = True
    __depends_on__ = ['Opportunity']
    
    # Template configuration as regular fields (these become columns in the generated data)
    __template_source__ = os.path.join(templates_dir, 'contract.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    # Fields needed for the template
    opportunity_id = None  # This will be filled in at runtime
    effective_date = None
    expiration_date = None
    contract_number = None
    customer_name = None
    customer_address = None
    service_description = None
    payment_terms = None
    contract_value = None
    renewal_terms = None
    legal_terms = None
    signatures = None
