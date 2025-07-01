"""
Tests for the unstructured module.
"""
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd

from syda.unstructured import UnstructuredDataProcessor


class TestUnstructuredDataProcessor:
    """Tests for the UnstructuredDataProcessor class."""
    
    def test_initialization(self):
        """Test initialization of UnstructuredDataProcessor."""
        processor = UnstructuredDataProcessor()
        
        # Check that supported types are properly initialized
        assert 'application/pdf' in processor.supported_types
        assert 'text/plain' in processor.supported_types
        assert 'text/html' in processor.supported_types
        assert 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in processor.supported_types
        assert 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in processor.supported_types
    
    @patch('syda.unstructured.magic.Magic')
    def test_get_file_type(self, mock_magic):
        """Test getting file type."""
        # Configure the mock
        mock_magic_instance = MagicMock()
        mock_magic_instance.from_file.return_value = 'text/plain'
        mock_magic.return_value = mock_magic_instance
        
        # Create processor and get file type
        processor = UnstructuredDataProcessor()
        file_type = processor._get_file_type('test.txt')
        
        # Check that the mock was called and the correct type was returned
        mock_magic_instance.from_file.assert_called_once_with('test.txt')
        assert file_type == 'text/plain'
    
    @patch('builtins.open', new_callable=mock_open, read_data='<!DOCTYPE html><html><body>Test</body></html>')
    def test_process_html(self, mock_file):
        """Test processing HTML files."""
        # Create processor and process HTML
        processor = UnstructuredDataProcessor()
        result = processor._process_html('test.html')
        
        # Check that the file was opened and the content was returned
        mock_file.assert_called_once_with('test.html', 'r', encoding='utf-8')
        assert 'text' in result
        assert result['text'] == '<!DOCTYPE html><html><body>Test</body></html>'
        assert result['type'] == 'html'
    
    @patch('builtins.open', side_effect=Exception('Test error'))
    def test_process_html_error(self, mock_file):
        """Test error handling when processing HTML files."""
        # Create processor and process HTML
        processor = UnstructuredDataProcessor()
        result = processor._process_html('test.html')
        
        # Check that the error was caught and returned
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['type'] == 'html'
    
    @patch('syda.unstructured.pytesseract')
    @patch('syda.unstructured.Image')
    def test_process_image(self, mock_image, mock_pytesseract):
        """Test processing image files."""
        # Configure the mocks
        mock_pytesseract.image_to_string.return_value = 'Extracted text from image'
        
        # Create processor and process image
        processor = UnstructuredDataProcessor()
        result = processor._process_image('test.png')
        
        # Check that OCR was called and the content was returned
        mock_pytesseract.image_to_string.assert_called_once()
        assert 'text' in result
        assert 'Extracted text from image' in result['text']
        assert result['type'] == 'image'
    
    @patch('syda.unstructured.Image.open')
    def test_process_image_error(self, mock_image_open):
        """Test error handling when processing image files."""
        # Configure mock to raise an exception
        mock_image_open.side_effect = Exception('Test error')
        
        # Create processor and process image
        processor = UnstructuredDataProcessor()
        result = processor._process_image('test.png')
        
        # Check that the error was caught and returned
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['type'] == 'image'
    
    @patch('syda.unstructured.pdfplumber.open')
    def test_process_pdf(self, mock_pdf_open):
        """Test processing PDF files."""
        # Configure the mocks
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = 'Page 1 content'
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        # Create processor and process PDF
        processor = UnstructuredDataProcessor()
        result = processor._process_pdf('test.pdf')
        
        # Check that PDF was processed and the content was returned
        mock_pdf_open.assert_called_once_with('test.pdf')
        mock_page.extract_text.assert_called_once()
        assert 'text' in result
        assert 'Page 1 content' in result['text']
        assert result['type'] == 'pdf'
    
    @patch('syda.unstructured.pdfplumber.open', side_effect=Exception('Test error'))
    def test_process_pdf_error(self, mock_pdf_open):
        """Test error handling when processing PDF files."""
        # Create processor and process PDF
        processor = UnstructuredDataProcessor()
        result = processor._process_pdf('test.pdf')
        
        # Check that the error was caught and returned
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['type'] == 'pdf'
    
    @patch('syda.unstructured.Document')
    def test_process_docx(self, mock_document_class):
        """Test processing DOCX files."""
        # Configure the mocks
        mock_document = MagicMock()
        mock_paragraph1 = MagicMock()
        mock_paragraph1.text = 'Paragraph 1'
        mock_paragraph2 = MagicMock()
        mock_paragraph2.text = 'Paragraph 2'
        mock_document.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_document_class.return_value = mock_document
        
        # Create processor and process DOCX
        processor = UnstructuredDataProcessor()
        result = processor._process_docx('test.docx')
        
        # Check that Document was initialized and the content was returned
        mock_document_class.assert_called_once_with('test.docx')
        assert 'text' in result
        assert 'Paragraph 1' in result['text']
        assert 'Paragraph 2' in result['text']
        assert result['type'] == 'docx'
    
    @patch('syda.unstructured.Document', side_effect=Exception('Test error'))
    def test_process_docx_error(self, mock_document_class):
        """Test error handling when processing DOCX files."""
        # Create processor and process DOCX
        processor = UnstructuredDataProcessor()
        result = processor._process_docx('test.docx')
        
        # Check that the error was caught and returned
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['type'] == 'docx'
    
    @patch('syda.unstructured.pd.read_excel')
    def test_process_excel(self, mock_read_excel):
        """Test processing Excel files."""
        # Configure the mocks
        mock_df = pd.DataFrame({'Column1': ['Value1', 'Value2'], 'Column2': ['Value3', 'Value4']})
        mock_read_excel.return_value = mock_df
        
        # Create processor and process Excel
        processor = UnstructuredDataProcessor()
        result = processor._process_excel('test.xlsx')
        
        # Check that read_excel was called and the content was returned
        mock_read_excel.assert_called_once_with('test.xlsx')
        assert result['type'] == 'excel'
        assert 'original_data' in result
        # Check that values are present in the original_data
        if isinstance(result['original_data'], list):
            # If converted to list of dicts
            assert len(result['original_data']) == 2
            assert 'Column1' in result['original_data'][0]
            assert 'Column2' in result['original_data'][0]
        else:
            # If kept as DataFrame
            assert 'Column1' in result['original_data'].columns
            assert 'Column2' in result['original_data'].columns
    
    @patch('syda.unstructured.pd.read_excel', side_effect=Exception('Test error'))
    def test_process_excel_error(self, mock_read_excel):
        """Test error handling when processing Excel files."""
        # Create processor and process Excel
        processor = UnstructuredDataProcessor()
        result = processor._process_excel('test.xlsx')
        
        # Check that the error was caught and returned
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['type'] == 'excel'
    
    @patch('builtins.open', new_callable=mock_open, read_data='Test text content\nMore text')
    def test_process_text(self, mock_file):
        """Test processing text files."""
        # Create processor and process text
        processor = UnstructuredDataProcessor()
        result = processor._process_text('test.txt')
        
        # Check that the file was opened and the content was returned
        mock_file.assert_called_once_with('test.txt', 'r', encoding='utf-8')
        assert 'text' in result
        assert result['text'] == 'Test text content\nMore text'
        assert result['type'] == 'text'
    
    @patch('builtins.open', side_effect=Exception('Test error'))
    def test_process_text_error(self, mock_file):
        """Test error handling when processing text files."""
        # Create processor and process text
        processor = UnstructuredDataProcessor()
        result = processor._process_text('test.txt')
        
        # Check that the error was caught and returned
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['type'] == 'text'
    
    @patch('syda.unstructured.UnstructuredDataProcessor._get_file_type')
    def test_process_file(self, mock_get_file_type):
        """Test processing a file with supported type."""
        # Configure the mock to return a supported file type
        mock_get_file_type.return_value = 'text/plain'
        
        # Create processor and mock the supported_types lookup
        processor = UnstructuredDataProcessor()
        mock_process_text = MagicMock(return_value={'text': 'Test content', 'type': 'text'})
        
        # Override the supported_types dict to use our mock
        original_supported_types = processor.supported_types
        processor.supported_types = {'text/plain': mock_process_text}
        
        try:
            # Process the file
            result = processor.process_file('test.txt')
            
            # Check that _get_file_type and the lookup were called correctly
            mock_get_file_type.assert_called_once_with('test.txt')
            mock_process_text.assert_called_once_with('test.txt')
            
            # Check that the result was returned
            assert result == {'text': 'Test content', 'type': 'text'}
        finally:
            # Restore the original supported_types
            processor.supported_types = original_supported_types
    
    @patch('syda.unstructured.UnstructuredDataProcessor._get_file_type')
    def test_process_file_unsupported_type(self, mock_get_file_type):
        """Test processing a file with unsupported type."""
        # Configure the mock
        mock_get_file_type.return_value = 'application/unknown'
        
        # Create processor and process file
        processor = UnstructuredDataProcessor()
        result = processor.process_file('test.unknown')
        
        # Check that _get_file_type was called
        mock_get_file_type.assert_called_once_with('test.unknown')
        
        # Check that an error was returned
        assert 'error' in result
        assert 'Unsupported file type' in result['error']
        assert result['type'] == 'application/unknown'
    
    def test_process_file_error(self):
        """Test error handling when processing a file."""
        # Create processor with mocked _get_file_type that raises an exception
        processor = UnstructuredDataProcessor()
        processor._get_file_type = MagicMock(side_effect=Exception('Test error'))
        
        # Process file with exception handling
        try:
            result = processor.process_file('test.txt')
            
            # This should not be reached as the method should propagate the exception
            assert False, "Expected exception was not raised"
        except Exception as e:
            # Check that the exception was raised with the expected message
            assert str(e) == 'Test error'
            
        # Check that _get_file_type was called
        processor._get_file_type.assert_called_once_with('test.txt')
    
    def test_generate_synthetic_file(self):
        """Test generate_synthetic_file method."""
        # Create processor
        processor = UnstructuredDataProcessor()
        
        # Create a mock handler function
        mock_handler = MagicMock(return_value={
            'text': 'Original content',
            'type': 'text'
        })
        
        # Save the original handlers
        original_handlers = processor.supported_types.copy()
        
        try:
            # Modify the supported_types dictionary to use our mock
            processor._get_file_type = MagicMock(return_value='text/plain')
            processor.supported_types['text/plain'] = mock_handler
            
            # Generate synthetic data
            result = processor.generate_synthetic_file('test.txt')
            
            # Check that our mock handler was called
            mock_handler.assert_called_once_with('test.txt')
            
            # Check that the error message about SyntheticDataGenerator is returned
            assert 'error' in result
            assert 'SyntheticDataGenerator class' in result['error']
            assert result['type'] == 'text/plain'
            assert result['original_data'] is None
        finally:
            # Restore the original handlers
            processor.supported_types = original_handlers
