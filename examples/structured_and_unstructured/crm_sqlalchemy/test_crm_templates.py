#!/usr/bin/env python3
"""
Example demonstrating the CRM synthetic data generation with template processing.

This example creates a simple CRM database with:
- Customers
- Contacts
- Opportunities
- Proposal documents (templates)
- Contract documents (templates)

It demonstrates how to use both structured database models and document templates together.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

# Import CRM data models
import models


def main():
    """Main entry point for the example."""
    print("Starting CRM template example...")
    
    # Initialize generator with model config (similar to retail example)
    config = ModelConfig(provider="anthropic", model_name="claude-3-5-haiku-20241022")
    generator = SyntheticDataGenerator(model_config=config)
    
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data for all models in one call with minimal custom generators
    results = generator.generate_for_sqlalchemy_models(
        sqlalchemy_models=[
            models.Customer,
            models.Contact,
            models.Opportunity,
            models.ProposalDocument, 
            models.ContractDocument
        ],
        sample_sizes={
            'customers': 5,
            'contacts': 10,
            'opportunities': 8,
            'proposal_documents': 3,
            'contract_documents': 2
        },
        output_dir=output_dir
    )
    

    
    # Print summary of generated data
    print("\n✅ Data generation complete!")
    for model_name, df in results.items():
        print(f"  {model_name}: {len(df)} records")
    
    print(f"\nData files saved to directory: {output_dir}")
    
    # Analyze randomness of parent schema selections
    print("\n🔍 Analyzing randomness of parent schema references:")
    
    # Check opportunity references in proposal_documents
    if 'proposal_documents' in results and 'opportunities' in results:
        proposal_df = results['proposal_documents']
        opp_df = results['opportunities']
        
        referenced_opps = {}
        for _, row in proposal_df.iterrows():
            opp_id = row.get('opportunity_id')
            if opp_id in referenced_opps:
                referenced_opps[opp_id] += 1
            else:
                referenced_opps[opp_id] = 1
        
        print("  proposal_documents references:")
        print(f"    - Unique opportunities referenced: {len(referenced_opps)} out of {len(proposal_df)} documents")
        print(f"    - Distribution: {referenced_opps}")
        
        # Check variety - are we using different opportunities?
        if len(referenced_opps) < len(proposal_df) and len(proposal_df) <= len(opp_df):
            print("    ⚠️  Some opportunities are referenced multiple times while others aren't used at all")
        elif len(referenced_opps) == 1 and len(opp_df) > 1:
            print("    ⚠️  All proposal_documents reference the same opportunity!")
        else:
            print("    ✅  Good variety in opportunity references")
    
    # Check opportunity references in contract_documents
    if 'contract_documents' in results and 'opportunities' in results:
        contract_df = results['contract_documents']
        opp_df = results['opportunities']
        
        referenced_opps = {}
        for _, row in contract_df.iterrows():
            opp_id = row.get('opportunity_id')
            if opp_id in referenced_opps:
                referenced_opps[opp_id] += 1
            else:
                referenced_opps[opp_id] = 1
        
        print("  contract_documents references:")
        print(f"    - Unique opportunities referenced: {len(referenced_opps)} out of {len(contract_df)} documents")
        print(f"    - Distribution: {referenced_opps}")
        
        # Check variety - are we using different opportunities?
        if len(referenced_opps) < len(contract_df) and len(contract_df) <= len(opp_df):
            print("    ⚠️  Some opportunities are referenced multiple times while others aren't used at all")
        elif len(referenced_opps) == 1 and len(opp_df) > 1:
            print("    ⚠️  All contract_documents reference the same opportunity!")
        else:
            print("    ✅  Good variety in opportunity references")
        
        # Check indirect customer references
        customer_ids = set()
        for opp_id in referenced_opps.keys():
            matching_opps = opp_df[opp_df['id'] == opp_id]
            if not matching_opps.empty:
                customer_ids.add(matching_opps.iloc[0]['customer_id'])
        
        print(f"    - Unique customers indirectly referenced: {len(customer_ids)}")
        if len(customer_ids) < len(referenced_opps) and len(referenced_opps) > 1:
            print("    ⚠️  Multiple opportunities from the same customer")
        elif len(customer_ids) == 1 and 'customers' in results and len(results['customers']) > 1:
            print("    ⚠️  All contract_documents reference opportunities from the same customer!")
        else:
            print("    ✅  Good variety in customer references")
    
    # Verify foreign key reference consistency
    print("\n🔍 Verifying data consistency within documents:")
    
    # Check if referential integrity is maintained
    print("\n🔍 Verifying referential integrity:")
    
    # Check proposal_documents consistency
    proposal_df = results.get('proposal_documents')
    opportunities_df = results.get('opportunities')
    customers_df = results.get('customers')
    
    if proposal_df is not None and opportunities_df is not None and customers_df is not None:
        # For each proposal document
        for idx, row in proposal_df.iterrows():
            proposal_opp_id = row.get('opportunity_id')
            proposal_opp_name = row.get('opportunity_name')
            proposal_opp_value = row.get('opportunity_value')
            proposal_opp_desc = row.get('opportunity_description')
            
            # Find the referenced opportunity
            opp_match = opportunities_df[opportunities_df['id'] == proposal_opp_id]
            if not opp_match.empty:
                opp_row = opp_match.iloc[0]
                # Check if all opportunity fields match
                name_match = proposal_opp_name == opp_row['name']
                value_match = proposal_opp_value == opp_row['value']
                desc_match = proposal_opp_desc == opp_row['description']
                
                if not (name_match and value_match and desc_match):
                    print(f"  ❌ Inconsistency in proposal_documents {idx+1} opportunity references:")
                    print(f"     - ID: {proposal_opp_id}")
                    print(f"     - Name match: {name_match}")
                    print(f"     - Value match: {value_match}")
                    print(f"     - Description match: {desc_match}")
                else:
                    print(f"  ✅ All proposal_documents.opportunity_id values reference valid opportunities.id")
                    print(f"  ✅ All proposal_documents.opportunity_name values reference valid opportunities.name")
                    print(f"  ✅ All proposal_documents.opportunity_value values reference valid opportunities.value")
                    print(f"  ✅ All proposal_documents.opportunity_description values reference valid opportunities.description")
            
            # Check customer consistency
            proposal_cust_name = row.get('customer_name')
            proposal_cust_addr = row.get('customer_address')
            
            # Find the referenced customer
            cust_match = customers_df[customers_df['name'] == proposal_cust_name]
            if not cust_match.empty:
                cust_row = cust_match.iloc[0]
                # Check if address matches
                addr_match = proposal_cust_addr == cust_row['address']
                
                if not addr_match:
                    print(f"  ❌ Inconsistency in proposal_documents {idx+1} customer references:")
                    print(f"     - Name: {proposal_cust_name}")
                    print(f"     - Address match: {addr_match}")
                else:
                    print(f"  ✅ All proposal_documents.customer_name values reference valid customers.name")
                    print(f"  ✅ All proposal_documents.customer_address values reference valid customers.address")
    
    # Check contract_documents consistency
    contract_df = results.get('contract_documents')
    if contract_df is not None and opportunities_df is not None and customers_df is not None:
        # For each contract document
        for idx, row in contract_df.iterrows():
            contract_opp_id = row.get('opportunity_id')
            contract_value = row.get('contract_value')
            
            # Find the referenced opportunity
            opp_match = opportunities_df[opportunities_df['id'] == contract_opp_id]
            if not opp_match.empty:
                opp_row = opp_match.iloc[0]
                # Check if contract value matches opportunity value
                value_match = contract_value == opp_row['value']
                
                if not value_match:
                    print(f"  ❌ Inconsistency in contract_documents {idx+1} opportunity references:")
                    print(f"     - ID: {contract_opp_id}")
                    print(f"     - Value match: {value_match}")
                else:
                    print(f"  ✅ All contract_documents.opportunity_id values reference valid opportunities.id")
                    print(f"  ✅ All contract_documents.contract_value values reference valid opportunities.value")
            
            # Check customer consistency
            contract_cust_name = row.get('customer_name')
            contract_cust_addr = row.get('customer_address')
            
            # Find the referenced customer
            cust_match = customers_df[customers_df['name'] == contract_cust_name]
            if not cust_match.empty:
                cust_row = cust_match.iloc[0]
                # Check if address matches
                addr_match = contract_cust_addr == cust_row['address']
                
                if not addr_match:
                    print(f"  ❌ Inconsistency in contract_documents {idx+1} customer references:")
                    print(f"     - Name: {contract_cust_name}")
                    print(f"     - Address match: {addr_match}")
                else:
                    print(f"  ✅ All contract_documents.customer_name values reference valid customers.name")
                    print(f"  ✅ All contract_documents.customer_address values reference valid customers.address")
            
    
    # Check if document templates were generated - fix the paths to match actual output
    output_path = Path(output_dir)
    
    # Look for document_*.pdf files in the ProposalDocument subdirectory
    proposal_dir = output_path / "proposal_documents"
    proposal_docs = list(proposal_dir.glob('document_*.pdf')) if proposal_dir.exists() else []
    
    # Look for document_*.pdf files in the ContractDocument subdirectory
    contract_dir = output_path / "contract_documents"
    contract_docs = list(contract_dir.glob('document_*.pdf')) if contract_dir.exists() else []
    
    print("\n📄 Generated Documents:")
    print(f"  - Proposals: {len(proposal_docs)} PDF documents")
    print(f"  - Contracts: {len(contract_docs)} PDF documents")


if __name__ == "__main__":
    main()
