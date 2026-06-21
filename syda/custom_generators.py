"""
Custom data generators for structured synthetic data generation.

This module provides a set of custom data generators that can be used
to generate synthetic data for structured data models.

The custom generators are functions that take in a row of data
and a column name, and return a generated value for that column.

The generators are registered with the GeneratorManager, which
provides a way to look up generators by column name.

"""


import random
import threading
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional, Union

class GeneratorManager:
    """
    Manages custom generators for data generation.

    The GeneratorManager is responsible for registering custom data generators
    for structured data models. It provides a way to look up generators by type
    or column name.

    The custom generators are functions that take in a row of data and a column
    name, and return a generated value for that column.

    The generators are registered with the GeneratorManager, which provides a way
    to look up generators by column name.

    The GeneratorManager is used by the SyntheticDataGenerator to generate
    synthetic data for structured data models.

    Args:
        type_name: The data type this generator handles
        func: The custom generator function
        column_name: The column name this generator should be registered for

    Attributes:
        type_generators: A dictionary mapping data types to custom generators
        column_generators: A dictionary mapping column names to custom generators
    """
    
    def __init__(self):
        """
        Initialize the generator manager.

        The GeneratorManager is responsible for registering custom data generators
        for structured data models. It provides a way to look up generators by type
        or column name.

        The custom generators are functions that take in a row of data and a column
        name, and return a generated value for that column.

        The generators are registered with the GeneratorManager, which provides a way
        to look up generators by column name.

        The GeneratorManager is used by the SyntheticDataGenerator to generate
        synthetic data for structured data models.
        """
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
            
        Returns:
            None
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
            
        Returns:
            None
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
            
        Returns:
            None
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
    
    def _register_consistent_fk_generators(
        self, 
        schema_name: str, 
        parent_schema: str, 
        parent_df: pd.DataFrame, 
        fk_list: List[Tuple[str, str]]
    ):
        """
        Register foreign key generators that ensure consistency across multiple columns 
        referencing the same parent table.
        
        Args:
            schema_name: Name of the schema being processed
            parent_schema: Name of the parent schema
            parent_df: DataFrame containing parent data
            fk_list: List of (fk_column, parent_column) tuples
            
        Returns:
            None
        """
        print(f"Ensuring consistent foreign keys for {len(fk_list)} columns in {schema_name} referencing {parent_schema}")
        
        # For each row we'll generate, select a consistent parent record index
        parent_indices = list(range(len(parent_df)))
        if not parent_indices:
            print(f"⚠️ Warning: No records in {parent_schema} for foreign keys in {schema_name}")
            return
        
        # Create a shared state between all generators for this table.
        # _lock guards row_cache to make the check-then-set atomic so that
        # two threads generating rows of the same table concurrently cannot
        # assign different parent records to the same row index.
        shared_state = {
            'parent_df': parent_df,
            'row_cache': {},
            'parent_indices': parent_indices,
            'parent_schema': parent_schema,
            '_lock': threading.Lock(),
        }
        
        # Register a generator for each column that uses the shared mapping.
        # Key is prefixed with schema name to avoid collisions when parallel tables
        # share a FK column name (e.g. two tables both have "customer_id").
        for fk_column, parent_column in fk_list:
            fk_generator = self._create_consistent_generator(fk_column, shared_state, parent_column)
            print(f"Registering consistent foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
            self.register_generator('foreign_key', fk_generator, column_name=f"{schema_name}__{fk_column}")
    
    def _create_consistent_generator(
        self, 
        col: str, 
        state: Dict, 
        parent_col: str
    ):
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
            if hasattr(row, 'name'):
                row_key = row.name
            else:
                row_key = hash(str(row))

            # Lock makes the check-then-set atomic so concurrent threads
            # generating different rows of the same table always pick a single
            # consistent parent record per row.
            with state['_lock']:
                if row_key not in state['row_cache']:
                    state['row_cache'][row_key] = {}
                if state['parent_schema'] not in state['row_cache'][row_key]:
                    parent_idx = random.choice(state['parent_indices']) \
                                 if state['parent_indices'] else 0
                    state['row_cache'][row_key][state['parent_schema']] = parent_idx
                parent_idx = state['row_cache'][row_key][state['parent_schema']]

            return state['parent_df'].iloc[parent_idx][parent_col]
        
        return generator
    
    def _register_simple_fk_generator(
        self, 
        schema_name: str, 
        parent_schema: str, 
        parent_df: pd.DataFrame, 
        fk_column: str, 
        parent_column: str
    ):
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
        
        # Register the generator for this column (schema-prefixed to prevent
        # collisions when parallel tables share a FK column name).
        print(f"Registering foreign key generator for {schema_name}.{fk_column} -> {parent_schema}.{parent_column}")
        self.register_generator('foreign_key', fk_generator, column_name=f"{schema_name}__{fk_column}")
        
    def apply_custom_generators(
        self, 
        df: pd.DataFrame, 
        model_name: str, 
        custom_generators: Dict, 
        parent_dfs: Optional[Dict] = None
    ) -> pd.DataFrame:
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
    
    def apply_type_generators(
        self,
        df: pd.DataFrame,
        llm_schema: Dict,
        schema_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply custom type-based and column-specific generators to the data.

        Args:
            df: DataFrame to apply generators to
            llm_schema: Dictionary mapping field names to types
            schema_name: Table name used to look up schema-prefixed FK generators.
                         Must match the prefix used during registration to avoid
                         parallel-table key collisions.

        Returns:
            DataFrame with generators applied
        """
        if df.empty:
            return df

        # Apply column-specific generators first (they take precedence).
        # FK generators are stored under "{schema_name}__{col}" keys; fall back
        # to bare col name for user-registered generators (no prefix).
        for col_name in df.columns:
            prefixed_key = f"{schema_name}__{col_name}" if schema_name else None
            key = prefixed_key if (prefixed_key and prefixed_key in self.column_generators) \
                  else (col_name if col_name in self.column_generators else None)
            if key:
                gen = self.column_generators[key]
                df[col_name] = df.apply(lambda row, g=gen: g(row, col_name), axis=1)

        # Then apply type-based generators
        for col_name, col_type in llm_schema.items():
            if col_name not in df.columns:
                continue
            # Skip if already handled by a column-specific generator
            prefixed_key = f"{schema_name}__{col_name}" if schema_name else None
            already_handled = (
                (prefixed_key and prefixed_key in self.column_generators)
                or col_name in self.column_generators
            )
            if already_handled:
                continue

            if isinstance(col_type, dict):
                base_type = col_type.get('type', 'text')
            else:
                base_type = col_type

            if base_type in self.type_generators:
                gen = self.type_generators[base_type]
                df[col_name] = df.apply(lambda row, g=gen: g(row, col_name), axis=1)

        return df
