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
            'ProposalDocument': 3,
            'ContractDocument': 2
        },
        output_dir=output_dir
    )
    
    # Print summary of generated data
    print("\n✅ Data generation complete!")
    for model_name, df in results.items():
        print(f"  {model_name}: {len(df)} records")
    
    print(f"\nData files saved to directory: {output_dir}")
    
    # Check if document templates were generated - fix the paths to match actual output
    output_path = Path(output_dir)
    
    # Look for document_*.pdf files in the ProposalDocument subdirectory
    proposal_dir = output_path / "ProposalDocument"
    proposal_docs = list(proposal_dir.glob('document_*.pdf')) if proposal_dir.exists() else []
    
    # Look for document_*.pdf files in the ContractDocument subdirectory
    contract_dir = output_path / "ContractDocument"
    contract_docs = list(contract_dir.glob('document_*.pdf')) if contract_dir.exists() else []
    
    print("\n📄 Generated Documents:")
    print(f"  - Proposals: {len(proposal_docs)} PDF documents")
    print(f"  - Contracts: {len(contract_docs)} PDF documents")


if __name__ == "__main__":
    main()
