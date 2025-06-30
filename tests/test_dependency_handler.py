"""
Tests for the dependency_handler module.
"""
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock

from syda.dependency_handler import DependencyHandler, ForeignKeyHandler


class TestDependencyHandler:
    """Tests for the DependencyHandler class."""
    
    def test_extract_dependencies(self):
        """Test extracting dependencies from foreign keys."""
        # Define test foreign keys
        foreign_keys = {
            "Order": {
                "customer_id": ("Customer", "id")
            },
            "OrderItem": {
                "order_id": ("Order", "id"),
                "product_id": ("Product", "id")
            },
            "Customer": {}  # No dependencies
        }
        
        # Extract dependencies
        dependencies = DependencyHandler.extract_dependencies(foreign_keys)
        
        # Check the result
        assert "Order" in dependencies
        assert "OrderItem" in dependencies
        assert "Customer" in dependencies
        assert dependencies["Order"] == ["Customer"]
        assert sorted(dependencies["OrderItem"]) == sorted(["Order", "Product"])
        assert dependencies["Customer"] == []
    
    def test_extract_dependencies_with_empty_input(self):
        """Test extracting dependencies from empty foreign keys."""
        # Extract dependencies from empty foreign keys
        dependencies = DependencyHandler.extract_dependencies({})
        
        # Check that the result is an empty dict
        assert dependencies == {}
    
    def test_build_dependency_graph(self):
        """Test building a dependency graph."""
        # Define test dependencies
        dependencies = {
            "Order": ["Customer"],
            "OrderItem": ["Order", "Product"],
            "Customer": []
        }
        
        # Build the dependency graph
        graph = DependencyHandler.build_dependency_graph(dependencies)
        
        # Check that the graph has the expected nodes and edges
        assert list(graph.nodes()) == ["Order", "OrderItem", "Customer", "Product"]
        assert ("Customer", "Order") in graph.edges()
        assert ("Order", "OrderItem") in graph.edges()
        assert ("Product", "OrderItem") in graph.edges()
    
    def test_determine_generation_order(self):
        """Test determining the generation order."""
        # Create a test graph
        graph = nx.DiGraph()
        graph.add_edge("Customer", "Order")  # Customer depends on Order
        graph.add_edge("Product", "OrderItem")  # Product depends on OrderItem
        graph.add_edge("Order", "OrderItem")  # Order depends on OrderItem
        
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
        # Define test dependencies with a cycle
        dependencies = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"]
        }
        
        # Build the dependency graph
        graph = DependencyHandler.build_dependency_graph(dependencies)
        
        # Check that the graph has the expected nodes and edges
        assert list(graph.nodes()) == ["A", "B", "C"]
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
        # Initialize the handler
        handler = ForeignKeyHandler()
        
        # Check that the attributes are properly initialized
        assert hasattr(handler, "dependencies")
        assert handler.dependencies == {}
    
    def test_register_dependencies(self):
        """Test registering dependencies."""
        # Create test dependencies
        dependencies = {
            "Order": ["Customer"],
            "OrderItem": ["Order", "Product"]
        }
        
        # Initialize the handler and register dependencies
        handler = ForeignKeyHandler()
        handler.register_dependencies(dependencies)
        
        # Check that the dependencies were registered
        assert handler.dependencies == dependencies
    
    def test_register_dataframes(self):
        """Test registering DataFrames."""
        # Create test DataFrames
        dataframes = {
            "Customer": MagicMock(),
            "Product": MagicMock()
        }
        
        # Initialize the handler and register DataFrames
        handler = ForeignKeyHandler()
        handler.register_dataframes(dataframes)
        
        # Check that the DataFrames were registered
        assert handler.dataframes == dataframes
    
    def test_resolve_foreign_key_dependencies(self):
        """Test resolving foreign key dependencies."""
        # Create a mock row
        row = {
            "customer_id": None,
            "product_id": None,
            "other_field": "value"
        }
        
        # Create mock DataFrames
        customer_df = MagicMock()
        customer_df.sample.return_value = MagicMock(iloc=[{"id": 42}])
        
        product_df = MagicMock()
        product_df.sample.return_value = MagicMock(iloc=[{"id": 100}])
        
        dataframes = {
            "Customer": customer_df,
            "Product": product_df
        }
        
        # Create test foreign keys
        foreign_keys = {
            "customer_id": ("Customer", "id"),
            "product_id": ("Product", "id")
        }
        
        # Initialize the handler and register DataFrames
        handler = ForeignKeyHandler()
        handler.register_dataframes(dataframes)
        
        # Resolve foreign key dependencies
        resolved_row = handler.resolve_foreign_key_dependencies(row, foreign_keys)
        
        # Check that the foreign keys were resolved
        assert resolved_row["customer_id"] == 42
        assert resolved_row["product_id"] == 100
        assert resolved_row["other_field"] == "value"
    
    def test_resolve_foreign_key_dependencies_with_empty_dataframe(self):
        """Test resolving foreign key dependencies with an empty DataFrame."""
        # Create a mock row
        row = {"customer_id": None}
        
        # Create a mock empty DataFrame
        customer_df = MagicMock()
        customer_df.empty = True
        
        dataframes = {"Customer": customer_df}
        
        # Create test foreign keys
        foreign_keys = {"customer_id": ("Customer", "id")}
        
        # Initialize the handler and register DataFrames
        handler = ForeignKeyHandler()
        handler.register_dataframes(dataframes)
        
        # Resolve foreign key dependencies
        with pytest.raises(ValueError, match="Referenced table .* is empty"):
            handler.resolve_foreign_key_dependencies(row, foreign_keys)
    
    def test_resolve_foreign_key_dependencies_with_missing_table(self):
        """Test resolving foreign key dependencies with a missing table."""
        # Create a mock row
        row = {"customer_id": None}
        
        # Create test foreign keys
        foreign_keys = {"customer_id": ("Customer", "id")}
        
        # Initialize the handler with empty dataframes
        handler = ForeignKeyHandler()
        handler.register_dataframes({})
        
        # Resolve foreign key dependencies
        with pytest.raises(ValueError, match="Referenced table .* not found"):
            handler.resolve_foreign_key_dependencies(row, foreign_keys)
