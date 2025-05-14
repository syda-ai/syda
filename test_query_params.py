#!/usr/bin/env python

"""
Simple test script to verify query parameter handling in proxy configuration
"""

import sys
import os

# Add the parent directory to the path so we can import the syda package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from syda.schemas import ProxyConfig

def test_query_params():
    # Test with various query parameter types
    proxy = ProxyConfig(
        base_url="https://ai-proxy.company.com/v1",
        params={
            "team": "data-science",
            "project": "synthetic-data",
            "track_usage": True, 
            "priority": 1,
            "tags": ["ai", "synthetic-data"]
        }
    )
    
    # Get the kwargs that would be used for client initialization
    kwargs = proxy.get_proxy_kwargs()
    
    # Print the result
    print("Proxy base_url with query parameters:")
    print(kwargs["base_url"])
    print("\nFull kwargs dictionary:")
    print(kwargs)
    
    # Test with URL that already has parameters
    proxy2 = ProxyConfig(
        base_url="https://ai-proxy.company.com/v1?api_version=2",
        params={"team": "engineering"}
    )
    
    kwargs2 = proxy2.get_proxy_kwargs()
    print("\nProxy with existing query parameters:")
    print(kwargs2["base_url"])

if __name__ == "__main__":
    test_query_params()
