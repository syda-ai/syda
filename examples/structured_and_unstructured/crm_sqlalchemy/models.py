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
    
    id = Column(Integer, primary_key=True, comment='Primary key for customer records')
    name = Column(String(100), nullable=False, comment='Full name of the customer organization')
    industry = Column(String(50), comment='Industry sector the customer operates in')
    annual_revenue = Column(Float, comment='Annual revenue of the customer organization in USD')
    employees = Column(Integer, comment='Total number of employees in the customer organization')
    website = Column(String(100), comment='Website URL of the customer organization')
    address = Column(String(200), comment='Physical street address of the customer headquarters')
    city = Column(String(50), comment='City where the customer headquarters is located')
    state = Column(String(2), comment='Two-letter state/province code')
    zip_code = Column(String(10), comment='Postal/zip code of the customer headquarters')
    status = Column(String(20), comment='Current status of the customer relationship (Active, Inactive, Prospect)')  # Active, Inactive, Prospect
    
    # Relationships
    contacts = relationship("Contact", back_populates="customer")
    opportunities = relationship("Opportunity", back_populates="customer")


class Contact(Base):
    """Individual person associated with a customer."""
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True, comment='Primary key for contact records')
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, comment='Foreign key reference to the customer this contact belongs to')
    first_name = Column(String(50), nullable=False, comment='Contact\'s first/given name')
    last_name = Column(String(50), nullable=False, comment='Contact\'s last/family name')
    email = Column(String(100), nullable=False, unique=True, comment='Contact\'s email address (unique across all contacts)')
    phone = Column(String(20), comment='Contact\'s phone number including country/area code')
    position = Column(String(100), comment='Job title or role within the customer organization')
    is_primary = Column(Boolean, comment='Whether this is the primary point of contact for the customer')
    
    # Relationships
    customer = relationship("Customer", back_populates="contacts")


class Opportunity(Base):
    """Sales opportunity with a customer."""
    __tablename__ = 'opportunities'
    
    id = Column(Integer, primary_key=True, comment='Primary key for opportunity records')
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, comment='Foreign key reference to the customer this opportunity is with')
    name = Column(String(100), nullable=False, comment='Name or title of the sales opportunity')
    value = Column(Float, nullable=False, comment='Estimated monetary value of the opportunity in USD')
    stage = Column(String(20), nullable=False, comment='Current stage in the sales pipeline (Lead, Qualification, Proposal, Negotiation, Closed Won, Closed Lost)')
    probability = Column(Float, comment='Estimated probability (0-100%) of winning the opportunity')
    expected_close_date = Column(Date, comment='Expected date when the opportunity will be closed (won or lost)')
    description = Column(Text, comment='Detailed description of the opportunity and customer requirements')
    
    # Relationships
    customer = relationship("Customer", back_populates="opportunities")

templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Template models that depend on structured data
class ProposalDocument(Base):
    """Sales proposal document for an opportunity."""
    # Special metadata attributes
    __tablename__ = 'proposal_documents'
    __depends_on__ = ['opportunities']
    
    # Template configuration as regular fields (these become columns in the generated data)
    __template__ = True
    __template_source__ = os.path.join(templates_dir, 'proposal.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True, comment='Primary key for proposal document records')
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False, comment='Foreign key reference to the opportunity this proposal is for')
    created_date = Column(Date, comment='Date when the proposal document was created')
    title = Column(String(200), comment='Main title of the proposal document')
    subtitle = Column(String(300), comment='Secondary title or tagline for the proposal')
    prepared_by = Column(String(100), comment='Name of the sales representative who prepared the proposal')
    customer_name = Column(String(100), ForeignKey('customers.name'), comment='Name of the customer organization (linked to customers table)')
    customer_address = Column(String(200), ForeignKey('customers.address'), comment='Address of the customer organization (linked to customers table)')
    opportunity_name = Column(String(100), ForeignKey('opportunities.name'), comment='Name of the opportunity (linked to opportunities table)')
    opportunity_value = Column(Float, ForeignKey('opportunities.value'), comment='Value of the opportunity in USD (linked to opportunities table)')
    opportunity_description = Column(Text, ForeignKey('opportunities.description'), comment='Description of the opportunity (linked to opportunities table)')
    proposed_solutions = Column(Text, comment='Detailed description of the proposed solutions/products/services')
    implementation_timeline = Column(Text, comment='Timeline for implementing the proposed solutions')
    pricing_details = Column(Text, comment='Detailed pricing information, including breakdowns and options')
    terms_and_conditions = Column(Text, comment='Standard terms and conditions for the proposal')


class ContractDocument(Base):
    """Contract document for a won opportunity."""
    # Special metadata attributes
    __tablename__ = 'contract_documents'
    __depends_on__ = ['opportunities']
    
    # Template configuration as regular fields (these become columns in the generated data)
    __template__ = True
    __template_source__ = os.path.join(templates_dir, 'contract.html')
    __input_file_type__ = 'html'
    __output_file_type__ = 'pdf'
    
    id = Column(Integer, primary_key=True, comment='Primary key for contract document records')
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), nullable=False, comment='Foreign key reference to the opportunity this contract is for')
    effective_date = Column(Date, comment='Date when the contract becomes effective')
    expiration_date = Column(Date, comment='Date when the contract expires')
    contract_number = Column(String(50), comment='Unique identifier/number for the contract')
    customer_name = Column(String(100), ForeignKey('customers.name'), comment='Name of the customer organization (linked to customers table)')
    customer_address = Column(String(200), ForeignKey('customers.address'), comment='Address of the customer organization (linked to customers table)')
    service_description = Column(Text, comment='Detailed description of services to be provided')
    payment_terms = Column(Text, comment='Payment terms including schedule, methods and conditions')
    contract_value = Column(Float, ForeignKey('opportunities.value'), comment='Total monetary value of the contract in USD (linked to opportunities table)')
    renewal_terms = Column(Text, comment='Terms for contract renewal or extension')
    legal_terms = Column(Text, comment='Legal terms and conditions including liabilities, warranties, etc.')
    signatures = Column(Text, comment='Signature blocks for both parties')