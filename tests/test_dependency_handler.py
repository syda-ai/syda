"""
Tests for the dependency_handler module.
"""
import pytest
import networkx as nx
import pandas as pd
from unittest.mock import patch, MagicMock

from syda.dependency_handler import DependencyHandler, ForeignKeyHandler


class TestDependencyHandler:
    """Tests for the DependencyHandler class."""
    
    def test_extract_dependencies(self):
        """Test extracting dependencies from foreign keys."""
        # Define test schemas
        schemas = {
            "Order": {"type": "object"},
            "OrderItem": {"type": "object"},
            "Customer": {"type": "object"},
            "Product": {"type": "object"}
        }
        
        # Define schema metadata
        schema_metadata = {
            "Order": {},
            "OrderItem": {},
            "Customer": {},
            "Product": {}
        }
        
        # Define test foreign keys
        foreign_keys = {
            "Order": {
                "customer_id": ("Customer", "id")
            },
            "OrderItem": {
                "order_id": ("Order", "id"),
                "product_id": ("Product", "id")
            },
            "Customer": {},  # No dependencies
            "Product": {}    # No dependencies
        }
        
        # Extract dependencies
        dependencies = DependencyHandler.extract_dependencies(schemas, schema_metadata, foreign_keys)
        
        # Check the result
        assert "Order" in dependencies
        assert "OrderItem" in dependencies
        assert "Customer" in dependencies
        assert dependencies["Order"] == ["Customer"]
        assert sorted(dependencies["OrderItem"]) == sorted(["Order", "Product"])
        assert dependencies["Customer"] == []
    
    def test_extract_dependencies_with_empty_input(self):
        """Test extracting dependencies from empty inputs."""
        # Define empty inputs
        schemas = {}
        schema_metadata = {}
        foreign_keys = {}
        
        # Extract dependencies from empty inputs
        dependencies = DependencyHandler.extract_dependencies(schemas, schema_metadata, foreign_keys)
        
        # Check that the result is an empty dict
        assert dependencies == {}
    
    def test_build_dependency_graph(self):
        """Test building a dependency graph."""
        # Define test nodes and dependencies
        nodes = ["Order", "OrderItem", "Customer", "Product"]
        dependencies = {
            "Order": ["Customer"],
            "OrderItem": ["Order", "Product"],
            "Customer": []
        }
        
        # Build the dependency graph
        graph = DependencyHandler.build_dependency_graph(nodes, dependencies)
        
        # Check that the graph has the expected nodes and edges
        assert sorted(list(graph.nodes())) == sorted(["Order", "OrderItem", "Customer", "Product"])
        assert ("Customer", "Order") in graph.edges()
        assert ("Order", "OrderItem") in graph.edges()
        assert ("Product", "OrderItem") in graph.edges()
    
    def test_determine_generation_order(self):
        """Test determining the generation order."""
        # Create a test graph
        graph = nx.DiGraph()
        # In dependency_handler.py, edges go from dependency to dependent
        # OrderItem is needed to generate Order
        graph.add_edge("OrderItem", "Order")  
        # OrderItem is needed to generate Product
        graph.add_edge("OrderItem", "Product")  
        # Order is needed to generate Customer
        graph.add_edge("Order", "Customer")  
        
        # Determine the generation order
        order = DependencyHandler.determine_generation_order(graph)
        
        # Check that the order is valid
        assert "OrderItem" in order
        assert "Order" in order
        assert "Customer" in order
        assert "Product" in order
        
        # Check that dependencies come before dependents
        assert order.index("OrderItem") < order.index("Order")
        assert order.index("OrderItem") < order.index("Product")
        assert order.index("Order") < order.index("Customer")
    
    def test_build_dependency_graph_with_cycle(self):
        """Test building a dependency graph with a cycle."""
        # Define test nodes and dependencies with a cycle
        nodes = ["A", "B", "C"]
        dependencies = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"]
        }
        
        # Build the dependency graph
        graph = DependencyHandler.build_dependency_graph(nodes, dependencies)
        
        # Check that the graph has the expected nodes and edges
        assert sorted(list(graph.nodes())) == sorted(["A", "B", "C"])
        assert ("B", "A") in graph.edges()
        assert ("C", "B") in graph.edges()
        assert ("A", "C") in graph.edges()
    
    def test_detect_cycles(self):
        """Test detecting cycles in a dependency graph."""
        # Create a test graph with a cycle
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        
        # Detect cycles
        has_cycle = DependencyHandler.has_cycle(graph)
        
        # Check that a cycle was detected
        assert has_cycle is True
        
        # Create a test graph without a cycle
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        # Detect cycles
        has_cycle = DependencyHandler.has_cycle(graph)
        
        # Check that no cycle was detected
        assert has_cycle is False


class TestForeignKeyHandler:
    """Tests for the ForeignKeyHandler class."""
    
    def test_initialization(self):
        """Test initializing the ForeignKeyHandler."""
        # Create a mock generator manager
        mock_generator_manager = MagicMock()
        
        # Initialize the handler with the mock generator manager
        handler = ForeignKeyHandler(generator_manager=mock_generator_manager)
        
        # Check that the generator_manager attribute is properly initialized
        assert hasattr(handler, "generator_manager")
        assert handler.generator_manager is mock_generator_manager
    
    def test_apply_foreign_keys(self):
        """Test applying foreign key constraints."""
        # Create a mock generator manager
        mock_generator_manager = MagicMock()
        
        # Initialize the handler with the mock generator manager
        handler = ForeignKeyHandler(generator_manager=mock_generator_manager)
        
        # Create test foreign keys
        extracted_foreign_keys = {
            "Order": {
                "customer_id": ("Customer", "id")
            }
        }
        
        # Create test results with dataframes
        customer_df = pd.DataFrame({"id": [1, 2, 3]})
        results = {"Customer": customer_df}
        
        # Apply foreign keys
        handler.apply_foreign_keys("Order", extracted_foreign_keys, results)
        
        # Check that the generator manager's register method was called
        # Even though Order is not in results, the method still registers a generator
        # for the foreign key since Customer is in results
        assert mock_generator_manager._register_simple_fk_generator.called
        
        # Verify the arguments passed to the register method
        mock_generator_manager._register_simple_fk_generator.assert_called_with(
            schema_name="Order",
            parent_schema="Customer",
            parent_df=customer_df,
            fk_column="customer_id",
            parent_column="id"
        )
    
    def test_verify_referential_integrity(self):
        """Test verifying referential integrity of generated data."""
        # Create a mock generator manager
        mock_generator_manager = MagicMock()
        
        # Initialize the handler with the mock generator manager
        handler = ForeignKeyHandler(generator_manager=mock_generator_manager)
        
        # Create test foreign keys
        extracted_foreign_keys = {
            "Order": {
                "customer_id": ("Customer", "id")
            }
        }
        
        # Create test results with valid references
        customer_df = pd.DataFrame({"id": [1, 2, 3]})
        order_df = pd.DataFrame({"customer_id": [1, 2, 3]})
        results = {"Customer": customer_df, "Order": order_df}
        
        # Verify referential integrity
        is_valid = handler.verify_referential_integrity(results, extracted_foreign_keys)
        
        # Check that the result is valid
        assert is_valid is True
    
    def test_verify_referential_integrity_with_invalid_references(self):
        """Test verifying referential integrity with invalid foreign key references."""
        # Create a mock generator manager
        mock_generator_manager = MagicMock()
        
        # Initialize the handler with the mock generator manager
        handler = ForeignKeyHandler(generator_manager=mock_generator_manager)
        
        # Create test foreign keys
        extracted_foreign_keys = {
            "Order": {
                "customer_id": ("Customer", "id")
            }
        }
        
        # Create test results with invalid references (customer_id=4 doesn't exist in Customer.id)
        customer_df = pd.DataFrame({"id": [1, 2, 3]})
        order_df = pd.DataFrame({"customer_id": [1, 2, 4]})  # 4 is invalid
        results = {"Customer": customer_df, "Order": order_df}
        
        # Verify referential integrity
        is_valid = handler.verify_referential_integrity(results, extracted_foreign_keys)
        
        # Check that the result is invalid due to the reference to non-existent id
        assert is_valid is False
    
    # End of TestForeignKeyHandler class
