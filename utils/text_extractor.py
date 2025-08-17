"""
Text extraction utilities for CV analysis
Supports PDF and image files
"""

import os
import re
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Optional, Tuple, List


class TextExtractor:
    """Extract text from PDF and image files"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from file based on its format
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text as string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self._extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                print(f"PyPDF2 also failed: {e2}")
                raise Exception(f"Failed to extract text from PDF: {e2}")
        
        return text.strip()
    
    def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Read image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not read image: {file_path}")
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(processed_image)
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Failed to extract text from image: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> dict:
        """
        Extract common CV sections
        
        Args:
            text: Raw CV text
            
        Returns:
            Dictionary with extracted sections
        """
        sections = {
            'contact': '',
            'education': '',
            'experience': '',
            'skills': '',
            'summary': ''
        }
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Define section keywords
        section_keywords = {
            'contact': ['contact', 'email', 'phone', 'address', 'linkedin'],
            'education': ['education', 'academic', 'degree', 'university', 'college', 'school'],
            'experience': ['experience', 'work history', 'employment', 'job', 'career'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise'],
            'summary': ['summary', 'objective', 'profile', 'about']
        }
        
        # Split text into lines
        lines = text.split('\n')
        
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line contains section keywords
            for section, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    current_section = section
                    break
            
            # Add line to current section
            if current_section and line.strip():
                sections[current_section] += line + '\n'
        
        # Clean up sections
        for section in sections:
            sections[section] = sections[section].strip()
        
        return sections


def extract_contact_info(text: str) -> dict:
    """Extract contact information from CV text"""
    contact_info = {
        'email': '',
        'phone': '',
        'linkedin': '',
        'address': ''
    }
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact_info['email'] = emails[0]
    
    # Phone pattern
    phone_pattern = r'(\+?[\d\s\-\(\)]{10,})'
    phones = re.findall(phone_pattern, text)
    if phones:
        contact_info['phone'] = phones[0]
    
    # LinkedIn pattern
    linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
    linkedin = re.findall(linkedin_pattern, text.lower())
    if linkedin:
        contact_info['linkedin'] = linkedin[0]
    
    return contact_info
