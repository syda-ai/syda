import networkx as nx
from typing import Dict, List, Tuple, Any, Set

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
    def extract_dependencies(cls, schemas: Dict, schema_metadata: Dict, foreign_keys: Dict) -> Dict:
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
            if isinstance(metadata_dict, dict) and '__depends_on__' in metadata_dict:
                explicit_deps = metadata_dict['__depends_on__']
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
