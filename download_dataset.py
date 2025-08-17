"""
Download CV datasets from Kaggle
"""

import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import json


def setup_kaggle():
    """Setup Kaggle API credentials"""
    # Check if kaggle.json exists
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_file):
        print("Kaggle API credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click on 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in ~/.kaggle/kaggle.json")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    return True


def download_resume_dataset():
    """Download resume dataset from Kaggle"""
    if not setup_kaggle():
        return False
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Download resume dataset
        print("Downloading Resume Dataset...")
        api.dataset_download_files(
            'snehaanbhawal/resume-dataset',
            path='data',
            unzip=True
        )
        
        print("Resume dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading resume dataset: {e}")
        return False


def download_job_posting_dataset():
    """Download job posting dataset from Kaggle"""
    if not setup_kaggle():
        return False
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Download job posting dataset
        print("Downloading Job Posting Dataset...")
        api.dataset_download_files(
            'promptcloud/job-posting-dataset',
            path='data',
            unzip=True
        )
        
        print("Job posting dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading job posting dataset: {e}")
        return False


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
        }
    ]
    
    # Create sample data directory
    os.makedirs('data/sample', exist_ok=True)
    
    # Save as JSON
    with open('data/sample/sample_cvs.json', 'w') as f:
        json.dump(sample_cvs, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame(sample_cvs)
    df.to_csv('data/sample/sample_cvs.csv', index=False)
    
    print("Sample CV data created successfully!")
    return True


def main():
    """Main function to download all datasets"""
    print("CV Analyzer - Dataset Downloader")
    print("=" * 40)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download datasets
    success_count = 0
    
    if download_resume_dataset():
        success_count += 1
    
    if download_job_posting_dataset():
        success_count += 1
    
    # Create sample data
    if create_sample_data():
        success_count += 1
    
    print(f"\nDownload completed! {success_count}/3 datasets processed successfully.")
    
    if success_count > 0:
        print("\nAvailable datasets:")
        if os.path.exists('data'):
            for root, dirs, files in os.walk('data'):
                level = root.replace('data', '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")


if __name__ == "__main__":
    main()
