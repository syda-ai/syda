"""
Output utilities for saving generated data to various formats.
"""

import os
import pandas as pd
from typing import Dict, Optional, Union, List


def save_dataframe(df: pd.DataFrame, file_path: str, format: Optional[str] = None) -> str:
    """
    Save a single DataFrame to a file with format detection and validation.
    
    Args:
        df: DataFrame to save
        file_path: Path where the file should be saved
        format: Optional format override ('csv' or 'json')
    
    Returns:
        Path to the saved file
    
    Raises:
        ValueError: If the DataFrame is empty or invalid
    """
    # Validate DataFrame
    if df.empty or len(df.columns) == 0:
        raise ValueError(
            "Failed to generate valid data. The resulting DataFrame is empty or has no columns. "
            "This could be due to an issue with the AI model response or schema definition. "
            "Check your schema, model settings, and API keys."
        )
    
    # Determine format from extension or override
    if format:
        # If format is explicitly provided, ensure path has correct extension
        if not file_path.endswith(f'.{format}'):
            file_path = f"{file_path}.{format}"
    
    # Save based on file extension
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False)
    elif file_path.endswith('.json'):
        df.to_json(file_path, orient='records')
    else:
        # Default to CSV if no recognized extension
        file_path = f"{file_path}.csv"
        df.to_csv(file_path, index=False)
    
    print(f"✓ Successfully wrote {len(df)} rows to {file_path}")
    return file_path


def save_dataframes(data_dict: Dict[str, pd.DataFrame], output_dir: str, format: str = 'csv') -> List[str]:
    """
    Save multiple DataFrames to files in a directory.
    
    Args:
        data_dict: Dictionary mapping names to DataFrames
        output_dir: Directory where files should be saved
        format: File format to use ('csv' or 'json')
    
    Returns:
        List of paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for name, df in data_dict.items():
        file_name = f"{name.lower()}.{format}"
        file_path = os.path.join(output_dir, file_name)
        saved_path = save_dataframe(df, file_path)
        saved_paths.append(saved_path)
    
    return saved_paths
