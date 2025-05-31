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
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False)  # This will be filled in at runtime
    created_date = Column(Date)
    title = Column(String(200))
    subtitle = Column(String(300))
    prepared_by = Column(String(100))
    customer_name = Column(String(100), ForeignKey('customers.name'))
    customer_address = Column(String(200), ForeignKey('customers.address'))
    opportunity_name = Column(String(100), ForeignKey('opportunities.name'))
    opportunity_value = Column(Float, ForeignKey('opportunities.value'))
    opportunity_description = Column(Text, ForeignKey('opportunities.description'))
    proposed_solutions = Column(Text)
    implementation_timeline = Column(Text)
    pricing_details = Column(Text)
    terms_and_conditions = Column(Text)


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
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False)
    effective_date = Column(Date)
    expiration_date = Column(Date)
    contract_number = Column(String(50))
    customer_name = Column(String(100), ForeignKey('customers.name'))
    customer_address = Column(String(200), ForeignKey('customers.address'))
    service_description = Column(Text)
    payment_terms = Column(Text)
    contract_value = Column(Float, ForeignKey('opportunities.value'))
    renewal_terms = Column(Text)
    legal_terms = Column(Text)
    signatures = Column(Text)