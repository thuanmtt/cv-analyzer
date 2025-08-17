"""
CV Analysis utilities
Includes scoring, recommendations, and content analysis
"""

import re
import nltk
import spacy
from textblob import TextBlob
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None


class CVAnalyzer:
    """Analyze CV content and provide scoring and recommendations"""
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # Define scoring weights
        self.weights = {
            'contact_info': 0.1,
            'education': 0.2,
            'experience': 0.3,
            'skills': 0.25,
            'summary': 0.15
        }
        
        # Define skill categories
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'mariadb'],
            'frameworks': ['django', 'flask', 'react', 'angular', 'vue', 'spring', 'express', 'fastapi'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'figma'],
            'languages': ['english', 'spanish', 'french', 'german', 'chinese', 'japanese']
        }
    
    def analyze_cv(self, text: str, sections: Dict[str, str]) -> Dict:
        """
        Comprehensive CV analysis
        
        Args:
            text: Raw CV text
            sections: Extracted CV sections
            
        Returns:
            Analysis results with scores and recommendations
        """
        analysis = {
            'overall_score': 0,
            'section_scores': {},
            'recommendations': [],
            'skill_analysis': {},
            'content_analysis': {},
            'contact_analysis': {}
        }
        
        # Analyze each section
        analysis['section_scores'] = {
            'contact_info': self._analyze_contact_info(sections.get('contact', '')),
            'education': self._analyze_education(sections.get('education', '')),
            'experience': self._analyze_experience(sections.get('experience', '')),
            'skills': self._analyze_skills(sections.get('skills', '')),
            'summary': self._analyze_summary(sections.get('summary', ''))
        }
        
        # Calculate overall score
        analysis['overall_score'] = self._calculate_overall_score(analysis['section_scores'])
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis['section_scores'], text)
        
        # Analyze skills
        analysis['skill_analysis'] = self._analyze_skill_distribution(text)
        
        # Content analysis
        analysis['content_analysis'] = self._analyze_content_quality(text)
        
        # Contact analysis
        analysis['contact_analysis'] = self._analyze_contact_completeness(sections.get('contact', ''))
        
        return analysis
    
    def _analyze_contact_info(self, contact_text: str) -> float:
        """Analyze contact information completeness"""
        score = 0
        contact_info = {
            'email': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', contact_text)),
            'phone': bool(re.search(r'(\+?[\d\s\-\(\)]{10,})', contact_text)),
            'linkedin': bool(re.search(r'linkedin\.com/in/[\w\-]+', contact_text.lower())),
            'name': len(contact_text.strip()) > 0
        }
        
        # Score based on completeness
        score = sum(contact_info.values()) / len(contact_info) * 100
        
        return min(score, 100)
    
    def _analyze_education(self, education_text: str) -> float:
        """Analyze education section"""
        if not education_text.strip():
            return 0
        
        score = 50  # Base score
        
        # Check for degree keywords
        degree_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma']
        degree_count = sum(1 for keyword in degree_keywords if keyword in education_text.lower())
        score += degree_count * 10
        
        # Check for university keywords
        university_keywords = ['university', 'college', 'institute', 'school']
        university_count = sum(1 for keyword in university_keywords if keyword in education_text.lower())
        score += university_count * 5
        
        # Check for graduation year
        year_pattern = r'\b(19|20)\d{2}\b'
        if re.search(year_pattern, education_text):
            score += 10
        
        return min(score, 100)
    
    def _analyze_experience(self, experience_text: str) -> float:
        """Analyze work experience section"""
        if not experience_text.strip():
            return 0
        
        score = 40  # Base score
        
        # Check for job titles
        job_titles = ['manager', 'director', 'lead', 'senior', 'junior', 'associate', 'analyst', 'engineer', 'developer']
        title_count = sum(1 for title in job_titles if title in experience_text.lower())
        score += title_count * 5
        
        # Check for company names (words with capital letters)
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        companies = re.findall(company_pattern, experience_text)
        score += min(len(companies) * 3, 20)
        
        # Check for dates/years
        date_pattern = r'\b(19|20)\d{2}\b'
        dates = re.findall(date_pattern, experience_text)
        score += min(len(dates) * 5, 15)
        
        # Check for action verbs
        action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'analyzed', 'coordinated']
        verb_count = sum(1 for verb in action_verbs if verb in experience_text.lower())
        score += verb_count * 2
        
        return min(score, 100)
    
    def _analyze_skills(self, skills_text: str) -> float:
        """Analyze skills section"""
        if not skills_text.strip():
            return 0
        
        score = 30  # Base score
        
        # Count technical skills
        all_skills = []
        for category, skills in self.skill_categories.items():
            all_skills.extend(skills)
        
        found_skills = sum(1 for skill in all_skills if skill in skills_text.lower())
        score += found_skills * 3
        
        # Check for skill categories
        category_count = sum(1 for category, skills in self.skill_categories.items() 
                           if any(skill in skills_text.lower() for skill in skills))
        score += category_count * 5
        
        return min(score, 100)
    
    def _analyze_summary(self, summary_text: str) -> float:
        """Analyze summary/objective section"""
        if not summary_text.strip():
            return 0
        
        score = 30  # Base score
        
        # Check length (not too short, not too long)
        word_count = len(summary_text.split())
        if 20 <= word_count <= 100:
            score += 20
        elif 10 <= word_count < 20 or 100 < word_count <= 150:
            score += 10
        
        # Check for professional keywords
        professional_keywords = ['experienced', 'skilled', 'passionate', 'dedicated', 'results-driven', 'team player']
        keyword_count = sum(1 for keyword in professional_keywords if keyword in summary_text.lower())
        score += keyword_count * 5
        
        # Check for industry keywords
        industry_keywords = ['technology', 'software', 'development', 'management', 'analysis', 'design']
        industry_count = sum(1 for keyword in industry_keywords if keyword in summary_text.lower())
        score += industry_count * 3
        
        return min(score, 100)
    
    def _calculate_overall_score(self, section_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        overall_score = 0
        
        for section, score in section_scores.items():
            weight = self.weights.get(section, 0.1)
            overall_score += score * weight
        
        return round(overall_score, 2)
    
    def _generate_recommendations(self, section_scores: Dict[str, float], text: str) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Contact info recommendations
        if section_scores['contact_info'] < 75:
            recommendations.append("Add complete contact information including email, phone, and LinkedIn profile")
        
        # Education recommendations
        if section_scores['education'] < 60:
            recommendations.append("Include more details about your education: degree, university, graduation year")
        
        # Experience recommendations
        if section_scores['experience'] < 70:
            recommendations.append("Add more detailed work experience with specific achievements and responsibilities")
        
        # Skills recommendations
        if section_scores['skills'] < 60:
            recommendations.append("Expand your skills section with more technical and soft skills")
        
        # Summary recommendations
        if section_scores['summary'] < 60:
            recommendations.append("Write a compelling professional summary (20-100 words)")
        
        # Content quality recommendations
        if len(text.split()) < 100:
            recommendations.append("Your CV seems too short. Consider adding more details about your experience and achievements")
        
        # Check for action verbs
        action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'analyzed']
        if not any(verb in text.lower() for verb in action_verbs):
            recommendations.append("Use more action verbs to describe your achievements")
        
        return recommendations
    
    def _analyze_skill_distribution(self, text: str) -> Dict:
        """Analyze skill distribution across categories"""
        skill_analysis = {}
        
        for category, skills in self.skill_categories.items():
            found_skills = [skill for skill in skills if skill in text.lower()]
            skill_analysis[category] = {
                'skills': found_skills,
                'count': len(found_skills),
                'percentage': len(found_skills) / len(skills) * 100
            }
        
        return skill_analysis
    
    def _analyze_content_quality(self, text: str) -> Dict:
        """Analyze overall content quality"""
        if not nlp:
            return {'error': 'spaCy model not available'}
        
        doc = nlp(text)
        
        # Calculate readability
        sentences = list(doc.sents)
        avg_sentence_length = np.mean([len(sent) for sent in sentences]) if sentences else 0
        
        # Calculate vocabulary diversity
        unique_words = set([token.text.lower() for token in doc if token.is_alpha])
        total_words = len([token.text for token in doc if token.is_alpha])
        vocabulary_diversity = len(unique_words) / total_words if total_words > 0 else 0
        
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        return {
            'word_count': total_words,
            'sentence_count': len(sentences),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'vocabulary_diversity': round(vocabulary_diversity, 3),
            'sentiment': round(sentiment, 3),
            'readability_score': self._calculate_readability(text)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = text.split('.')
        words = text.split()
        syllables = self._count_syllables(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return max(0, min(100, score))
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (approximate)"""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return count
    
    def _analyze_contact_completeness(self, contact_text: str) -> Dict:
        """Analyze contact information completeness"""
        contact_info = {
            'email': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', contact_text)),
            'phone': bool(re.search(r'(\+?[\d\s\-\(\)]{10,})', contact_text)),
            'linkedin': bool(re.search(r'linkedin\.com/in/[\w\-]+', contact_text.lower())),
            'name': len(contact_text.strip()) > 0
        }
        
        completeness_score = sum(contact_info.values()) / len(contact_info) * 100
        
        return {
            'completeness_score': round(completeness_score, 2),
            'missing_fields': [field for field, present in contact_info.items() if not present],
            'contact_details': contact_info
        }
