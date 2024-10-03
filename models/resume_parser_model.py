
import re
import spacy
from spacy.matcher import Matcher
from typing import List, Dict

class ResumeParser:
    def __init__(self):
        """
        Initializes the ResumeParser with the necessary NLP models and resources.
        """
        # Load the spaCy English model
        self.nlp = spacy.load('en_core_web_sm')
        # Initialize the Matcher
        self.matcher = Matcher(self.nlp.vocab)
        # Add patterns to the matcher
        self.add_patterns()
        # Load skills from file
        self.skills = self.load_skills('data/skills.txt')

    def load_skills(self, skills_file: str) -> List[str]:
        """
        Loads a list of skills from a text file.

        Args:
            skills_file (str): Path to the skills text file.

        Returns:
            List[str]: A list of skills.
        """
        with open(skills_file, 'r') as f:
            skills = [line.strip().lower() for line in f]
        return skills

    def add_patterns(self):
        """
        Adds custom patterns to the spaCy Matcher for entity extraction.
        """
        # Example pattern for extracting education degrees
        degree_pattern = [
            {'LOWER': 'bachelor'}, {'LOWER': 'of'}, {'LOWER': 'science'},
            {'LOWER': 'in'}, {'IS_ALPHA': True, 'OP': '+'}
        ]
        self.matcher.add('DEGREE', [degree_pattern])

    def extract_name(self, text: str) -> str:
        """
        Extracts the name from the resume text.

        Args:
            text (str): The resume text.

        Returns:
            str: The extracted name.
        """
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                return ent.text
        return None

    def extract_contact_info(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts contact information from the resume text.

        Args:
            text (str): The resume text.

        Returns:
            Dict[str, List[str]]: A dictionary containing emails and phone numbers.
        """
        email_pattern = r'[\w\.-]+@[\w\.-]+'
        phone_pattern = r'\+?\d[\d -]{8,12}\d'

        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)

        return {'emails': emails, 'phones': phones}

    def extract_education(self, text: str) -> List[str]:
        """
        Extracts education details from the resume text.

        Args:
            text (str): The resume text.

        Returns:
            List[str]: A list of education qualifications.
        """
        doc = self.nlp(text)
        education = []

        for sent in doc.sents:
            if 'education' in sent.text.lower():
                education.append(sent.text.strip())

        # Alternatively, use the Matcher for more complex patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            education.append(span.text)

        return education

    def extract_skills(self, text: str) -> List[str]:
        """
        Extracts skills from the resume text.

        Args:
            text (str): The resume text.

        Returns:
            List[str]: A list of skills.
        """
        doc = self.nlp(text.lower())
        skillset = []

        for token in doc:
            if token.text in self.skills:
                skillset.append(token.text)

        # Remove duplicates
        skillset = list(set(skillset))

        return skillset

    def parse(self, text: str) -> Dict[str, any]:
        """
        Parses the resume text and extracts relevant information.

        Args:
            text (str): The resume text.

        Returns:
            Dict[str, any]: A dictionary containing extracted information.
        """
        name = self.extract_name(text)
        contact_info = self.extract_contact_info(text)
        education = self.extract_education(text)
        skills = self.extract_skills(text)

        return {
            'name': name,
            'contact_info': contact_info,
            'education': education,
            'skills': skills,
            'text': text  # Including the text for further processing if needed
        }
