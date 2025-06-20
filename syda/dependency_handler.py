import networkx as nx
from typing import Dict, List, Tuple, Any, Set
import pandas as pd


class ForeignKeyHandler:
    """
    Handles foreign key operations for the data generation process.
    
    This class encapsulates all operations related to foreign keys, including:
    - Applying foreign key constraints during data generation
    - Verifying referential integrity of generated data
    """
    
    def __init__(self, generator_manager):
        """
        Initialize the ForeignKeyHandler.
        
        Args:
            generator_manager: Generator manager instance for registering foreign key generators
        """
        self.generator_manager = generator_manager
    
    def apply_foreign_keys(self, schema_name, extracted_foreign_keys, results):
        """
        Apply foreign key constraints to the specified schema.
        
        This method registers appropriate generators for foreign key columns to ensure
        that foreign key relationships are maintained in the generated data.
        
        Args:
            schema_name: Name of the schema being processed
            extracted_foreign_keys: Dictionary of foreign key definitions
            results: Dictionary of dataframes with already generated data
            
        Returns:
            None
        """
        # Apply foreign key constraints if applicable
        if schema_name not in extracted_foreign_keys:
            return
            
        # Group foreign keys by parent table
        fk_by_parent = {}
        for fk_column, (parent_schema, parent_column) in extracted_foreign_keys[schema_name].items():
            if parent_schema not in fk_by_parent:
                fk_by_parent[parent_schema] = []
            fk_by_parent[parent_schema].append((fk_column, parent_column))
        
        # Process each parent table group
        for parent_schema, fk_list in fk_by_parent.items():
            if parent_schema in results:
                parent_df = results[parent_schema]
                
                # Multiple columns referencing the same parent table
                if len(fk_list) > 1:
                    print(f"Ensuring consistent foreign keys for {len(fk_list)} columns in {schema_name} referencing {parent_schema}")
                    
                    # Get the list of column pairs for registration
                    column_pairs = [(fk_column, parent_column) for fk_column, parent_column in fk_list]
                    
                    # Register consistent foreign key generators for these columns
                    parent_indices = list(range(len(parent_df)))
                    if not parent_indices:
                        print(f"⚠️ Warning: No records in {parent_schema} for foreign keys in {schema_name}")
                        continue
                    
                    # Register all consistent foreign key generators at once
                    print(f"Registering consistent foreign key generators for {schema_name} -> {parent_schema}")
                    self.generator_manager._register_consistent_fk_generators(
                        schema_name=schema_name,
                        parent_schema=parent_schema,
                        parent_df=parent_df,
                        fk_list=fk_list
                    )
                else:
                    # Only one column referencing this parent table, use simple generator
                    for fk_column, parent_column in fk_list:
                        valid_values = parent_df[parent_column].tolist()
                        
                        if not valid_values:
                            print(f"⚠️ Warning: No valid values found in {parent_schema}.{parent_column} for foreign key {schema_name}.{fk_column}")
                            continue
                        
                        # Register a simple foreign key generator
                        print(f"Registering foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
                        self.generator_manager._register_simple_fk_generator(
                            schema_name=schema_name,
                            parent_schema=parent_schema,
                            parent_df=parent_df,
                            fk_column=fk_column,
                            parent_column=parent_column
                        )
            else:
                for fk_column, parent_column in fk_list:
                    print(f"⚠️ Warning: Parent schema {parent_schema} not available for foreign key {schema_name}.{fk_column}")
    
    def verify_referential_integrity(self, results, extracted_foreign_keys):
        """
        Verify that all foreign key relationships are valid in the generated data.
        
        Args:
            results: Dictionary of dataframes with generated data
            extracted_foreign_keys: Dictionary of foreign key definitions
            
        Returns:
            Boolean indicating if all foreign key relationships are valid
        """
        print("\n🔍 Verifying referential integrity:")
        
        all_valid = True
        
        for schema_name, fk_dict in extracted_foreign_keys.items():
            if schema_name not in results:
                continue
                
            df = results[schema_name]
            
            for fk_column, (parent_schema, parent_column) in fk_dict.items():
                if parent_schema not in results:
                    print(f"  ⚠️ Warning: Parent schema {parent_schema} not found for {schema_name}.{fk_column}")
                    all_valid = False
                    continue
                    
                parent_df = results[parent_schema]
                
                if fk_column not in df.columns:
                    print(f"  ⚠️ Warning: Foreign key column {fk_column} not found in {schema_name}")
                    all_valid = False
                    continue
                    
                if parent_column not in parent_df.columns:
                    print(f"  ⚠️ Warning: Referenced column {parent_column} not found in {parent_schema}")
                    all_valid = False
                    continue
                    
                # Get all values in the foreign key column
                fk_values = df[fk_column].dropna().unique()
                
                # Get all values in the parent column
                parent_values = set(parent_df[parent_column].unique())
                
                # Check if all foreign key values exist in the parent column
                invalid_values = [v for v in fk_values if v not in parent_values]
                
                if invalid_values:
                    print(f"  ❌ Error: Found {len(invalid_values)} invalid references in {schema_name}.{fk_column} to {parent_schema}.{parent_column}")
                    print(f"     Invalid values: {invalid_values[:5]}{'...' if len(invalid_values) > 5 else ''}")
                    all_valid = False
                else:
                    print(f"  ✅ All {schema_name}.{fk_column} values reference valid {parent_schema}.{parent_column}")
                    
        return all_valid


class DependencyHandler:
    """Handles dependency resolution for related schemas."""
    
    @staticmethod
    def build_dependency_graph(nodes, dependencies):
        """
        Build a directed graph of dependencies.
        
        Args:
            nodes: List of node names to add to the graph
            dependencies: Dict mapping node names to their dependencies
            
        Returns:
            NetworkX DiGraph representing dependencies between nodes
        """
        # Create a directed graph
        graph = nx.DiGraph()
        
        # Add all nodes to the graph
        for node in nodes:
            graph.add_node(node)
            
        # Add dependency edges (from dependency to dependent)
        for node, deps in dependencies.items():
            if node not in graph:
                graph.add_node(node)
                
            for dep in deps:
                if dep not in graph:
                    graph.add_node(dep)
                # Add edge from dependency to dependent
                graph.add_edge(dep, node)
                
        return graph
    
    @classmethod
    def extract_dependencies(
        cls, 
        schemas: Dict, 
        schema_metadata: Dict, 
        foreign_keys: Dict, 
        schema_depends_on_schemas: Dict = {}
    ) -> Dict:
        """
        Extract all dependencies from schemas, metadata, and foreign keys.
        
        Args:
            schemas: Dictionary of schema definitions
            schema_metadata: Dictionary of schema metadata
            foreign_keys: Dictionary mapping schema names to their foreign key definitions
            
        Returns:
            Dictionary mapping schema names to lists of dependencies
        """
        all_dependencies = {schema_name: [] for schema_name in schemas.keys()}
        
        # Extract explicit dependencies from metadata
        for schema_name, metadata_dict in schema_metadata.items():
            if isinstance(metadata_dict, dict) and schema_name in schema_depends_on_schemas:
                explicit_deps = schema_depends_on_schemas[schema_name]
                if isinstance(explicit_deps, list):
                    for dep in explicit_deps:
                        if dep not in all_dependencies[schema_name]:
                            all_dependencies[schema_name].append(dep)
                elif isinstance(explicit_deps, str):
                    if explicit_deps not in all_dependencies[schema_name]:
                        all_dependencies[schema_name].append(explicit_deps)
        
        # Add foreign key dependencies
        for schema_name, fk_columns in foreign_keys.items():
            for fk_column, (parent_schema, parent_column) in fk_columns.items():
                if parent_schema not in all_dependencies[schema_name]:
                    all_dependencies[schema_name].append(parent_schema)
        
        return all_dependencies
    
    @staticmethod
    def determine_generation_order(dependency_graph):
        """
        Determine the optimal generation order based on a dependency graph.
        
        Args:
            dependency_graph: NetworkX DiGraph representing schema dependencies
            
        Returns:
            List of schema names in optimal generation order
        """
        try:
            return list(nx.topological_sort(dependency_graph))
        except nx.NetworkXUnfeasible:
            # If there's a cycle in the graph, we can't sort topologically
            # Fall back to using nodes in arbitrary order
            print(f"Warning: Cycle detected in dependency graph. Using arbitrary order.")
            return list(dependency_graph.nodes())
        except Exception as e:
            print(f"Warning: Could not determine optimal generation order: {str(e)}")
            return list(dependency_graph.nodes())
