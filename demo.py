"""
Demo script for CV Analyzer
Test the functionality with sample data
"""

import os
import json
from utils.text_extractor import TextExtractor, extract_contact_info
from utils.cv_analyzer import CVAnalyzer


def demo_text_extraction():
    """Demo text extraction functionality"""
    print("🔍 Testing Text Extraction...")
    
    text_extractor = TextExtractor()
    
    # Test with sample CV text
    sample_cv_text = """
    JOHN DOE
    Software Engineer
    john.doe@email.com | +1-555-0123 | linkedin.com/in/johndoe
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development. Passionate about creating scalable web applications using modern technologies.
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology, 2018
    
    EXPERIENCE
    Senior Software Engineer
    TechCorp (2020-2023)
    - Led development of microservices architecture
    - Managed team of 5 developers
    - Implemented CI/CD pipelines
    
    SKILLS
    Python, JavaScript, React, Django, PostgreSQL, AWS, Docker, Git
    """
    
    # Clean text
    cleaned_text = text_extractor.clean_text(sample_cv_text)
    print(f"✅ Text cleaned successfully")
    
    # Extract sections
    sections = text_extractor.extract_sections(cleaned_text)
    print(f"✅ Sections extracted: {list(sections.keys())}")
    
    # Extract contact info
    contact_info = extract_contact_info(cleaned_text)
    print(f"✅ Contact info extracted: {contact_info}")
    
    return cleaned_text, sections


def demo_cv_analysis():
    """Demo CV analysis functionality"""
    print("\n📊 Testing CV Analysis...")
    
    cv_analyzer = CVAnalyzer()
    
    # Get sample data
    cleaned_text, sections = demo_text_extraction()
    
    # Analyze CV
    analysis_results = cv_analyzer.analyze_cv(cleaned_text, sections)
    
    print(f"✅ Overall Score: {analysis_results['overall_score']:.1f}")
    print(f"✅ Section Scores:")
    for section, score in analysis_results['section_scores'].items():
        print(f"   - {section}: {score:.1f}%")
    
    print(f"✅ Recommendations: {len(analysis_results['recommendations'])}")
    for i, rec in enumerate(analysis_results['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    return analysis_results


def demo_sample_data():
    """Demo with sample data from file"""
    print("\n📁 Testing with Sample Data...")
    
    sample_file = "data/sample/sample_cvs.json"
    if not os.path.exists(sample_file):
        print(f"❌ Sample data file not found: {sample_file}")
        return
    
    with open(sample_file, 'r') as f:
        sample_cvs = json.load(f)
    
    cv_analyzer = CVAnalyzer()
    
    print(f"📊 Analyzing {len(sample_cvs)} sample CVs...")
    
    for i, cv in enumerate(sample_cvs, 1):
        # Combine CV sections
        text = f"{cv.get('summary', '')} {cv.get('experience', '')} {cv.get('skills', '')}"
        
        # Extract sections
        sections = {
            'contact': f"{cv.get('name', '')} {cv.get('email', '')} {cv.get('phone', '')}",
            'education': cv.get('education', ''),
            'experience': cv.get('experience', ''),
            'skills': cv.get('skills', ''),
            'summary': cv.get('summary', '')
        }
        
        # Analyze
        analysis = cv_analyzer.analyze_cv(text, sections)
        
        print(f"CV {i} ({cv.get('name', 'Unknown')}): {analysis['overall_score']:.1f}")

def demo_skill_analysis():
    """Demo skill analysis functionality"""
    print("\n🔧 Testing Skill Analysis...")
    
    cv_analyzer = CVAnalyzer()
    
    # Test with different skill sets
    test_cases = [
        {
            'name': 'Software Engineer',
            'skills': 'Python, JavaScript, React, Django, PostgreSQL, AWS, Docker, Git'
        },
        {
            'name': 'Data Scientist',
            'skills': 'Python, R, SQL, TensorFlow, Scikit-learn, Tableau, AWS, Jupyter'
        },
        {
            'name': 'DevOps Engineer',
            'skills': 'AWS, Docker, Kubernetes, Jenkins, Terraform, Ansible, Linux, Python'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['name']}:")
        skill_analysis = cv_analyzer._analyze_skill_distribution(test_case['skills'])
        
        for category, data in skill_analysis.items():
            if data['skills']:
                print(f"   {category.title()}: {', '.join(data['skills'])} ({data['count']} skills)")


def main():
    """Main demo function"""
    print("🚀 CV Analyzer Demo")
    print("=" * 50)
    
    try:
        # Run demos
        demo_text_extraction()
        demo_cv_analysis()
        demo_sample_data()
        demo_skill_analysis()
        
        print("\n✅ All demos completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Run 'streamlit run app.py' to start the web application")
        print("2. Upload a CV file to test the full functionality")
        print("3. Run 'python process_archive_data.py' to process your dataset")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check that all dependencies are installed correctly.")


if __name__ == "__main__":
    main()
