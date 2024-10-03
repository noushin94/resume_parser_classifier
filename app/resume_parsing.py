
import os
from PyPDF2 import PdfReader
from models.resume_parser_model import ResumeParser

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def parse_resume(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    parser = ResumeParser()
    parsed_data = parser.parse(text)
    return parsed_data
