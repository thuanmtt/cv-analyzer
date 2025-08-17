"""
Process CV dataset from archive folder
"""

import os
import json
import csv
from utils.text_extractor import TextExtractor
from utils.cv_analyzer import CVAnalyzer


def process_resume_csv():
    """Process the Resume.csv file from archive"""
    csv_path = "archive/Resume/Resume.csv"
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return False
    
    print("Processing Resume.csv...")
    
    # Initialize analyzers
    text_extractor = TextExtractor()
    cv_analyzer = CVAnalyzer()
    
    processed_data = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for i, row in enumerate(reader):
                if i >= 50:  # Process only first 50 for demo
                    break
                
                # Extract text from resume
                resume_text = row.get('Resume', '')
                if not resume_text:
                    continue
                
                # Clean text
                cleaned_text = text_extractor.clean_text(resume_text)
                
                # Extract sections
                sections = text_extractor.extract_sections(cleaned_text)
                
                # Analyze CV
                analysis = cv_analyzer.analyze_cv(cleaned_text, sections)
                
                # Create processed record
                processed_record = {
                    'id': i + 1,
                    'category': row.get('Category', 'Unknown'),
                    'resume_text': cleaned_text,
                    'sections': sections,
                    'analysis': analysis,
                    'overall_score': analysis['overall_score'],
                    'contact_info': extract_contact_info(cleaned_text)
                }
                
                processed_data.append(processed_record)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} resumes...")
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        
        with open('data/processed/processed_resumes.json', 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Successfully processed {len(processed_data)} resumes")
        print("Data saved to data/processed/processed_resumes.json")
        
        # Create summary statistics
        create_summary_stats(processed_data)
        
        return True
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return False


def create_summary_stats(processed_data):
    """Create summary statistics from processed data"""
    if not processed_data:
        return
    
    # Calculate statistics
    scores = [item['overall_score'] for item in processed_data]
    categories = [item['category'] for item in processed_data]
    
    stats = {
        'total_resumes': len(processed_data),
        'average_score': sum(scores) / len(scores),
        'min_score': min(scores),
        'max_score': max(scores),
        'categories': list(set(categories)),
        'category_counts': {}
    }
    
    # Count categories
    for category in categories:
        stats['category_counts'][category] = stats['category_counts'].get(category, 0) + 1
    
    # Save statistics
    with open('data/processed/summary_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Summary statistics saved to data/processed/summary_stats.json")
    
    # Print summary
    print("\n📊 Summary Statistics:")
    print(f"Total resumes processed: {stats['total_resumes']}")
    print(f"Average score: {stats['average_score']:.2f}")
    print(f"Score range: {stats['min_score']:.1f} - {stats['max_score']:.1f}")
    print(f"Categories found: {len(stats['categories'])}")
    print("Category distribution:")
    for category, count in stats['category_counts'].items():
        print(f"  - {category}: {count}")


def extract_contact_info(text):
    """Extract contact information from text"""
    from utils.text_extractor import extract_contact_info
    return extract_contact_info(text)


def main():
    """Main function"""
    print("CV Analyzer - Archive Data Processor")
    print("=" * 40)
    
    # Process resume data
    if process_resume_csv():
        print("\n✅ Data processing completed successfully!")
    else:
        print("\n❌ Data processing failed!")


if __name__ == "__main__":
    main()
