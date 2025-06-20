import random
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional, Union

class GeneratorManager:
    """Manages custom generators for data generation."""
    
    def __init__(self):
        """Initialize the generator manager."""
        # Registry for custom generators by type: type_name -> fn(row: pd.Series, col_name: str) -> value
        self.type_generators: Dict[str, Callable[[pd.Series, str], any]] = {}
        
        # Registry for custom generators by column name: col_name -> fn(row: pd.Series, col_name: str) -> value
        self.column_generators: Dict[str, Callable[[pd.Series, str], any]] = {}
    
    def register_generator(self, type_name: str, func: Callable[[pd.Series, str], Any], 
                           column_name: Optional[str] = None):
        """
        Register a custom generator for a specific data type or column name.
        
        Args:
            type_name: The data type this generator handles (e.g., 'number', 'text', 'foreign_key')
            func: Function that takes (row: pd.Series, col_name: str) and returns a generated value
            column_name: If specified, this generator only applies to the named column
                        rather than all columns of the specified type
        """
        if column_name:
            self.column_generators[column_name] = func
        else:
            self.type_generators[type_name] = func
    
    def get_generator_state(self):
        """
        Get a copy of the current generator state.
        
        Returns:
            Tuple of (type_generators, column_generators)
        """
        return (self.type_generators.copy(), self.column_generators.copy())
    
    def restore_generator_state(self, state):
        """
        Restore generator state from a previous backup.
        
        Args:
            state: Tuple of (type_generators, column_generators)
        """
        self.type_generators, self.column_generators = state
    
    def register_foreign_key_generators(self, schema_name: str, fk_columns: Dict, results: Dict, sample_size: int):
        """
        Register appropriate foreign key generators for a schema.
        
        Args:
            schema_name: Name of the schema being processed
            fk_columns: Dictionary mapping foreign key columns to (parent_schema, parent_column) tuples
            results: Dictionary of already generated dataframes
            sample_size: Number of records to generate for this schema
        """
        if not fk_columns:
            return
            
        # Group foreign keys by parent table for more efficient processing
        fk_by_parent = {}
        for fk_column, (parent_schema, parent_column) in fk_columns.items():
            if parent_schema not in fk_by_parent:
                fk_by_parent[parent_schema] = []
            fk_by_parent[parent_schema].append((fk_column, parent_column))
        
        # Process each parent table group
        for parent_schema, fk_list in fk_by_parent.items():
            if parent_schema not in results:
                for fk_column, parent_column in fk_list:
                    print(f"⚠️ Warning: Parent schema {parent_schema} not available for foreign key {schema_name}.{fk_column}")
                continue
                
            parent_df = results[parent_schema]
            
            # Multiple columns referencing the same parent table - ensure consistency
            if len(fk_list) > 1:
                self._register_consistent_fk_generators(schema_name, parent_schema, parent_df, fk_list)
            else:
                # Only one column referencing this parent table, use regular random selection
                for fk_column, parent_column in fk_list:
                    self._register_simple_fk_generator(schema_name, parent_schema, parent_df, fk_column, parent_column)
    
    def _register_consistent_fk_generators(self, schema_name, parent_schema, parent_df, fk_list):
        """
        Register foreign key generators that ensure consistency across multiple columns 
        referencing the same parent table.
        
        Args:
            schema_name: Name of the schema being processed
            parent_schema: Name of the parent schema
            parent_df: DataFrame containing parent data
            fk_list: List of (fk_column, parent_column) tuples
        """
        print(f"Ensuring consistent foreign keys for {len(fk_list)} columns in {schema_name} referencing {parent_schema}")
        
        # For each row we'll generate, select a consistent parent record index
        parent_indices = list(range(len(parent_df)))
        if not parent_indices:
            print(f"⚠️ Warning: No records in {parent_schema} for foreign keys in {schema_name}")
            return
        
        # Create a shared state between all generators
        shared_state = {
            'parent_df': parent_df,
            'row_cache': {},  # Cache to store row-to-parent mappings
            'parent_indices': parent_indices,
            'parent_schema': parent_schema
        }
        
        # Register a generator for each column that uses the shared mapping
        for fk_column, parent_column in fk_list:
            fk_generator = self._create_consistent_generator(fk_column, shared_state, parent_column)
            print(f"Registering consistent foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
            self.register_generator('foreign_key', fk_generator, column_name=fk_column)
    
    def _create_consistent_generator(self, col, state, parent_col):
        """
        Create a generator function that produces consistent foreign key values
        for a given row across multiple columns referencing the same parent.
        
        Args:
            col: Foreign key column name
            state: Shared state dictionary
            parent_col: Parent column name
            
        Returns:
            Generator function that takes (row, col_name) and returns a value
        """
        def generator(row, col_name):
            # Get a stable identifier for this row
            if hasattr(row, 'name'):
                row_key = row.name  # Pandas row index
            else:
                row_key = hash(str(row))
            
            # Two-level cache: by row and parent table
            if row_key not in state['row_cache']:
                state['row_cache'][row_key] = {}
                
            # If we haven't assigned a parent for this row and this parent table
            if state['parent_schema'] not in state['row_cache'][row_key]:
                # Select a random parent index
                if state['parent_indices']:
                    parent_idx = random.choice(state['parent_indices'])
                else:
                    parent_idx = 0  # Fallback if no parent indices
                    
                # Store in cache
                state['row_cache'][row_key][state['parent_schema']] = parent_idx
                
            # Get the parent index for this row and parent table
            parent_idx = state['row_cache'][row_key][state['parent_schema']]
            
            # Return the value from the parent record
            return state['parent_df'].iloc[parent_idx][parent_col]
        
        return generator
    
    def _register_simple_fk_generator(self, schema_name, parent_schema, parent_df, fk_column, parent_column):
        """
        Register a simple foreign key generator that randomly selects from valid parent values.
        
        Args:
            schema_name: Name of the schema being processed
            parent_schema: Name of the parent schema
            parent_df: DataFrame containing parent data
            fk_column: Foreign key column name
            parent_column: Parent column name
        """
        valid_values = parent_df[parent_column].tolist()
        
        if not valid_values:
            print(f"⚠️ Warning: No valid values found in {parent_schema}.{parent_column} for foreign key {schema_name}.{fk_column}")
            return
        
        # Create a generator that returns a random valid value
        values_copy = valid_values.copy()  # Make a copy to avoid reference issues
        fk_generator = lambda row, col, values=values_copy: random.choice(values)
        
        # Register the generator for this column
        print(f"Registering foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
        self.register_generator('foreign_key', fk_generator, column_name=fk_column)
        
    def apply_custom_generators(self, df: pd.DataFrame, model_name: str, 
                               custom_generators: Dict, parent_dfs: Optional[Dict] = None):
        """
        Apply custom generators to the generated data.
        
        Args:
            df: DataFrame to apply generators to
            model_name: Name of the model being processed
            custom_generators: Dictionary of custom generators for the model
            parent_dfs: Optional dictionary of previously generated dataframes
            
        Returns:
            DataFrame with custom generators applied
        """
        if not df.empty and custom_generators:
            for col_name, generator in custom_generators.items():
                # Skip if column doesn't exist
                if col_name not in df.columns:
                    continue
                    
                # Apply the generator to each row
                try:
                    if parent_dfs is not None:
                        # Pass in parent dataframes if generator accepts them
                        if generator.__code__.co_argcount >= 3:
                            df[col_name] = df.apply(lambda row: generator(row, col_name, parent_dfs), axis=1)
                        else:
                            df[col_name] = df.apply(lambda row: generator(row, col_name), axis=1)
                    else:
                        df[col_name] = df.apply(lambda row: generator(row, col_name), axis=1)
                        
                except Exception as e:
                    print(f"Error applying custom generator for {model_name}.{col_name}: {str(e)}")
        
        return df
    
    def apply_type_generators(self, df: pd.DataFrame, llm_schema: Dict):
        """
        Apply custom type-based and column-specific generators to the data.
        
        Args:
            df: DataFrame to apply generators to
            llm_schema: Dictionary mapping field names to types
            
        Returns:
            DataFrame with generators applied
        """
        if df.empty:
            return df
            
        # Apply column-specific generators first (they take precedence)
        for col_name in df.columns:
            if col_name in self.column_generators:
                df[col_name] = df.apply(
                    lambda row: self.column_generators[col_name](row, col_name), 
                    axis=1
                )
        
        # Then apply type-based generators
        for col_name, col_type in llm_schema.items():
            # Skip if column doesn't exist or already handled by column-specific generator
            if col_name not in df.columns or col_name in self.column_generators:
                continue
                
            # Get base type (handle dict definitions)
            if isinstance(col_type, dict):
                base_type = col_type.get('type', 'text')
            else:
                base_type = col_type
                
            # Apply type generator if available
            if base_type in self.type_generators:
                df[col_name] = df.apply(
                    lambda row: self.type_generators[base_type](row, col_name), 
                    axis=1
                )
        
        return df
