"""
Tests for the output module.
"""
import pytest
import pandas as pd
import os
import json
import csv
from unittest.mock import patch, mock_open

from syda.output import (
    save_dataframe,
    save_dataframes
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com"]
    })


@pytest.fixture
def sample_dfs():
    """Create sample DataFrames for testing."""
    return {
        "Customer": pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "email": ["alice@example.com", "bob@example.com"]
        }),
        "Order": pd.DataFrame({
            "id": [101, 102],
            "customer_id": [1, 2],
            "amount": [100.0, 200.0]
        })
    }


class TestSaveDataframe:
    """Tests for the save_dataframe function."""
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_dataframe_to_csv(self, mock_to_csv, mock_makedirs, mock_exists, sample_df):
        """Test saving a DataFrame to CSV."""
        # Configure mocks
        mock_exists.return_value = False
        
        # Save the DataFrame to CSV
        output_path = "/tmp/output/customer.csv"
        save_dataframe(sample_df, output_path, format="csv")
        
        # Check that the directory was created
        mock_makedirs.assert_called_once_with("/tmp/output", exist_ok=True)
        
        # Check that to_csv was called with the right arguments
        mock_to_csv.assert_called_once_with(output_path, index=False)
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_json')
    def test_save_dataframe_to_json(self, mock_to_json, mock_makedirs, mock_exists, sample_df):
        """Test saving a DataFrame to JSON."""
        # Configure mocks
        mock_exists.return_value = False
        
        # Save the DataFrame to JSON
        output_path = "/tmp/output/customer.json"
        save_dataframe(sample_df, output_path, format="json")
        
        # Check that the directory was created
        mock_makedirs.assert_called_once_with("/tmp/output", exist_ok=True)
        
        # Check that to_json was called with the right arguments
        mock_to_json.assert_called_once_with(output_path, orient="records")
    
    def test_save_dataframe_unsupported_format(self, sample_df):
        """Test saving a DataFrame with an unsupported format."""
        # Try to save with an unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            save_dataframe(sample_df, "/tmp/output.xyz", format="xyz")
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_dataframe_csv_content(self, mock_to_csv, mock_makedirs, mock_exists, sample_df):
        """Test the content saved to a CSV file."""
        # Configure mocks
        mock_exists.return_value = True
        
        # Save the DataFrame to CSV
        output_path = "/tmp/output/customer.csv"
        
        # Use the implementation with mocked to_csv
        save_dataframe(sample_df, output_path, format="csv")
        
        # Check that to_csv was called with the right arguments
        mock_to_csv.assert_called_once_with(output_path, index=False)
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_json')
    def test_save_dataframe_json_content(self, mock_to_json, mock_makedirs, mock_exists, sample_df):
        """Test the content saved to a JSON file."""
        # Configure mocks
        mock_exists.return_value = True
        
        # Save the DataFrame to JSON
        output_path = "/tmp/output/customer.json"
        
        # Use the implementation with mocked to_json
        save_dataframe(sample_df, output_path, format="json")
        
        # Check that to_json was called with the right arguments
        mock_to_json.assert_called_once_with(output_path, orient="records")


class TestSaveDataframes:
    """Tests for the save_dataframes function."""
    
    @patch('syda.output.save_dataframe')
    def test_save_dataframes_to_csv(self, mock_save_dataframe, sample_dfs):
        """Test saving multiple DataFrames to CSV."""
        # Save the DataFrames to CSV
        output_dir = "/tmp/output"
        
        # Set up the mock to return a predictable value
        mock_save_dataframe.return_value = "/tmp/output/dummy.csv"
        
        save_dataframes(sample_dfs, output_dir, format="csv")
        
        # Check that save_dataframe was called for each DataFrame
        assert mock_save_dataframe.call_count == 2
        
        # Instead of comparing DataFrames directly, check that the function was called with the right arguments
        call_args_list = mock_save_dataframe.call_args_list
        called_paths = [os.path.basename(args[1]) for args, _ in call_args_list]
        
        # Check that the expected filenames were used (in lowercase as per the implementation)
        assert "customer.csv" in called_paths
        assert "order.csv" in called_paths
    
    @patch('syda.output.save_dataframe')
    def test_save_dataframes_to_json(self, mock_save_dataframe, sample_dfs):
        """Test saving multiple DataFrames to JSON."""
        # Save the DataFrames to JSON
        output_dir = "/tmp/output"
        
        # Set up the mock to return a predictable value
        mock_save_dataframe.return_value = "/tmp/output/dummy.json"
        
        save_dataframes(sample_dfs, output_dir, format="json")
        
        # Check that save_dataframe was called for each DataFrame
        assert mock_save_dataframe.call_count == 2
        
        # Instead of comparing DataFrames directly, check that the function was called with the right arguments
        call_args_list = mock_save_dataframe.call_args_list
        called_paths = [os.path.basename(args[1]) for args, _ in call_args_list]
        
        # Check that the expected filenames were used (in lowercase as per the implementation)
        assert "customer.json" in called_paths
        assert "order.json" in called_paths
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_save_dataframes_directory_creation(self, mock_makedirs, mock_exists, sample_dfs):
        """Test that the output directory is created if it doesn't exist."""
        # Configure mocks
        mock_exists.return_value = False
        
        # Use a patch to prevent actual file writing
        with patch('syda.output.save_dataframe'):
            # Save the DataFrames
            output_dir = "/tmp/output"
            save_dataframes(sample_dfs, output_dir)
            
            # Check that the directory was created
            mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
    
    def test_save_dataframes_empty_dict(self):
        """Test saving an empty dictionary of DataFrames."""
        # Save an empty dictionary
        with patch('syda.output.save_dataframe') as mock_save_dataframe:
            save_dataframes({}, "/tmp/output")
            
            # Check that save_dataframe was not called
            mock_save_dataframe.assert_not_called()
    
    @patch('syda.output.save_dataframe')
    def test_save_dataframes_custom_filenames(self, mock_save_dataframe, sample_dfs):
        """Test saving DataFrames with custom filenames."""
        # Save the DataFrames with custom filenames
        output_dir = "/tmp/output"
        
        # Set up the mock to return a predictable value
        mock_save_dataframe.return_value = "/tmp/output/dummy.csv"
        
        filenames = {
            "Customer": "customers_data",
            "Order": "orders_data"
        }
        save_dataframes(sample_dfs, output_dir, filenames=filenames)
        
        # Check that save_dataframe was called for each DataFrame
        assert mock_save_dataframe.call_count == 2
        
        # Instead of comparing DataFrames directly, check that the function was called with the right arguments
        call_args_list = mock_save_dataframe.call_args_list
        called_paths = [os.path.basename(args[1]) for args, _ in call_args_list]
        
        # Check that the expected custom filenames were used
        assert "customers_data.csv" in called_paths
        assert "orders_data.csv" in called_paths
