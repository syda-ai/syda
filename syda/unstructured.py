import os
import magic
import io
import re
from PIL import Image
import pytesseract
import pdfplumber
from docx import Document
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Set, Tuple

class UnstructuredDataProcessor:
    def __init__(self):
        self.supported_types = {
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._process_excel,
            'text/plain': self._process_text
        }

    def _get_file_type(self, file_path: str) -> str:
        """Get the MIME type of a file"""
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)

    def _process_image(self, file_path: str) -> Dict:
        """Process image files (JPEG, PNG)"""
        try:
            # Extract text from image using OCR
            text = pytesseract.image_to_string(Image.open(file_path))
            
            # Create DataFrame for processing
            df = pd.DataFrame({'text': [text]})
            
            return {
                'text': text,
                'type': 'image'
            }
        except Exception as e:
            return {'error': str(e), 'type': 'image'}

    def _process_pdf(self, file_path: str) -> Dict:
        """Process PDF files"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
            
            # Create DataFrame for processing
            df = pd.DataFrame({'text': [text]})
            
            return {
                'text': text,
                'type': 'pdf'
            }
        except Exception as e:
            return {'error': str(e), 'type': 'pdf'}

    def _process_docx(self, file_path: str) -> Dict:
        """Process Word documents (.docx)"""
        try:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            
            # Create DataFrame for processing
            df = pd.DataFrame({'text': [text]})
            
            return {
                'text': text,
                'type': 'docx'
            }
        except Exception as e:
            return {'error': str(e), 'type': 'docx'}

    def _process_excel(self, file_path: str) -> Dict:
        """Process Excel files (.xlsx)"""
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            return {
                'original_data': df.to_dict(orient='records'),
                'type': 'excel'
            }
        except Exception as e:
            return {'error': str(e), 'type': 'excel'}

    def _process_text(self, file_path: str) -> Dict:
        """Process plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create DataFrame for processing
            df = pd.DataFrame({'text': [text]})
            
            return {
                'text': text,
                'type': 'text'
            }
        except Exception as e:
            return {'error': str(e), 'type': 'text'}

    def process_file(self, file_path: str) -> Dict:
        """Process any supported file type"""
        file_type = self._get_file_type(file_path)
        
        if file_type not in self.supported_types:
            return {
                'error': f"Unsupported file type: {file_type}",
                'type': file_type
            }
        
        return self.supported_types[file_type](file_path)

    def generate_synthetic_file(self, file_path: str, n_samples: int = 100) -> Dict:
        """Generate synthetic data based on file content"""
        file_type = self._get_file_type(file_path)
        
        if file_type not in self.supported_types:
            return {
                'error': f"Unsupported file type: {file_type}",
                'type': file_type
            }
        
        # First process the file to get its structure
        result = self.supported_types[file_type](file_path)
        
        if 'error' in result:
            return result
        
        # For now, we don't support direct synthetic data generation in this class
        # This functionality has been moved to the SyntheticDataGenerator class
        return {
            'error': "Synthetic data generation now requires using the SyntheticDataGenerator class",
            'type': file_type,
            'original_data': result.get('original_data', None)
        }
