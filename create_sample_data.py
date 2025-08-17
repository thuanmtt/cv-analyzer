"""
Create sample CV data for testing
"""

import os
import json


def create_sample_data():
    """Create sample CV data for testing"""
    sample_cvs = [
        {
            'name': 'John Doe',
            'email': 'john.doe@email.com',
            'phone': '+1-555-0123',
            'linkedin': 'linkedin.com/in/johndoe',
            'summary': 'Experienced software engineer with 5+ years in full-stack development. Passionate about creating scalable web applications using modern technologies.',
            'education': 'Bachelor of Science in Computer Science, University of Technology, 2018',
            'experience': 'Senior Software Engineer at TechCorp (2020-2023)\n- Led development of microservices architecture\n- Managed team of 5 developers\nSoftware Developer at StartupInc (2018-2020)\n- Developed REST APIs using Python and Django',
            'skills': 'Python, JavaScript, React, Django, PostgreSQL, AWS, Docker, Git',
            'score': 85
        },
        {
            'name': 'Jane Smith',
            'email': 'jane.smith@email.com',
            'phone': '+1-555-0456',
            'linkedin': 'linkedin.com/in/janesmith',
            'summary': 'Data scientist with expertise in machine learning and statistical analysis. Dedicated to solving complex business problems through data-driven insights.',
            'education': 'Master of Science in Data Science, Data University, 2020',
            'experience': 'Data Scientist at AnalyticsCo (2020-2023)\n- Built predictive models for customer behavior\n- Implemented A/B testing frameworks\nData Analyst at ResearchLab (2018-2020)\n- Conducted statistical analysis on large datasets',
            'skills': 'Python, R, SQL, TensorFlow, Scikit-learn, Tableau, AWS, Jupyter',
            'score': 92
        },
        {
            'name': 'Mike Johnson',
            'email': 'mike.johnson@email.com',
            'phone': '+1-555-0789',
            'linkedin': 'linkedin.com/in/mikejohnson',
            'summary': 'Product manager with strong background in agile methodologies and user experience design. Results-driven leader focused on delivering customer value.',
            'education': 'Bachelor of Business Administration, Business School, 2017',
            'experience': 'Product Manager at ProductCo (2019-2023)\n- Launched 3 successful products\n- Managed $2M product budget\nAssociate PM at StartupXYZ (2017-2019)\n- Conducted user research and market analysis',
            'skills': 'Product Management, Agile, Scrum, User Research, Figma, SQL, Jira, A/B Testing',
            'score': 78
        },
        {
            'name': 'Sarah Wilson',
            'email': 'sarah.wilson@email.com',
            'phone': '+1-555-0321',
            'linkedin': 'linkedin.com/in/sarahwilson',
            'summary': 'UX/UI designer with 4 years of experience creating user-centered digital experiences. Passionate about accessibility and user research.',
            'education': 'Bachelor of Design, Design Institute, 2019',
            'experience': 'Senior UX Designer at DesignStudio (2021-2023)\n- Led design system implementation\n- Conducted user research and usability testing\nUX Designer at CreativeAgency (2019-2021)\n- Designed mobile and web applications\n- Collaborated with development teams',
            'skills': 'Figma, Sketch, Adobe Creative Suite, User Research, Prototyping, Design Systems, Accessibility',
            'score': 88
        },
        {
            'name': 'David Chen',
            'email': 'david.chen@email.com',
            'phone': '+1-555-0654',
            'linkedin': 'linkedin.com/in/davidchen',
            'summary': 'DevOps engineer with expertise in cloud infrastructure and automation. Committed to improving deployment processes and system reliability.',
            'education': 'Bachelor of Engineering, Engineering University, 2018',
            'experience': 'DevOps Engineer at CloudCorp (2020-2023)\n- Implemented CI/CD pipelines\n- Managed Kubernetes clusters\nSystem Administrator at TechStartup (2018-2020)\n- Maintained server infrastructure\n- Automated deployment processes',
            'skills': 'AWS, Docker, Kubernetes, Jenkins, Terraform, Ansible, Linux, Python, Bash',
            'score': 91
        }
    ]
    
    # Create sample data directory
    os.makedirs('data/sample', exist_ok=True)
    
    # Save as JSON
    with open('data/sample/sample_cvs.json', 'w') as f:
        json.dump(sample_cvs, f, indent=2)
    
    print("Sample CV data created successfully!")
    print(f"Created {len(sample_cvs)} sample CVs in data/sample/sample_cvs.json")
    
    return True


def create_sample_pdf_cv():
    """Create a sample PDF CV for testing"""
    # This would require a PDF library like reportlab
    # For now, we'll create a simple text file that can be used for testing
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
- Optimized database queries improving performance by 40%

Software Developer
StartupInc (2018-2020)
- Developed REST APIs using Python and Django
- Built responsive user interfaces with React
- Participated in code reviews and technical planning

SKILLS
Programming: Python, JavaScript, Java, C++
Frameworks: Django, React, Spring Boot, Express
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS, Docker, Kubernetes
Tools: Git, Jenkins, Jira, VS Code
"""
    
    # Create sample CV text file
    os.makedirs('data/sample', exist_ok=True)
    with open('data/sample/sample_cv.txt', 'w') as f:
        f.write(sample_cv_text)
    
    print("Sample CV text file created: data/sample/sample_cv.txt")
    return True


if __name__ == "__main__":
    print("Creating sample data for CV Analyzer...")
    create_sample_data()
    create_sample_pdf_cv()
    print("\nSample data creation completed!")
