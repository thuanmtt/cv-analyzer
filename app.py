"""
CV Analyzer - Streamlit Application
Main application for CV analysis and evaluation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
from datetime import datetime
import base64

# Import custom modules
from utils.text_extractor import TextExtractor, extract_contact_info
from utils.cv_analyzer import CVAnalyzer

# Page configuration
st.set_page_config(
    page_title="CV Analyzer - AI-Powered Resume Evaluation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .skill-category {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📄 CV Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Resume Evaluation System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Upload CV")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CV file",
            type=['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            help="Upload your CV in PDF or image format"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # Save uploaded file temporarily
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Analyze button
            if st.button("🔍 Analyze CV", type="primary"):
                with st.spinner("Analyzing your CV..."):
                    analyze_cv(f"temp_{uploaded_file.name}")
        
        st.markdown("---")
        st.header("📊 Features")
        st.markdown("""
        - **Text Extraction**: PDF & Image support
        - **AI Analysis**: Transformer-based evaluation
        - **Scoring System**: Comprehensive scoring algorithm
        - **Recommendations**: Personalized improvement tips
        - **Visualization**: Interactive charts and graphs
        """)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This CV Analyzer uses advanced AI techniques to evaluate resume quality and provide actionable recommendations for improvement.
        """)
    
    # Main content area
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)
    else:
        display_welcome()

def analyze_cv(file_path):
    """Analyze uploaded CV"""
    try:
        # Initialize extractors and analyzers
        text_extractor = TextExtractor()
        cv_analyzer = CVAnalyzer()
        
        # Extract text
        raw_text = text_extractor.extract_text(file_path)
        cleaned_text = text_extractor.clean_text(raw_text)
        
        # Extract sections
        sections = text_extractor.extract_sections(cleaned_text)
        
        # Analyze CV
        analysis_results = cv_analyzer.analyze_cv(cleaned_text, sections)
        
        # Add extracted text and sections to results
        analysis_results['raw_text'] = raw_text
        analysis_results['cleaned_text'] = cleaned_text
        analysis_results['sections'] = sections
        analysis_results['contact_info'] = extract_contact_info(cleaned_text)
        
        # Store results in session state
        st.session_state.analysis_results = analysis_results
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        st.error(f"Error analyzing CV: {str(e)}")
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

def display_welcome():
    """Display welcome message and instructions"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 Welcome to CV Analyzer!
        
        **Upload your CV** to get started with AI-powered analysis:
        
        ### ✅ What we analyze:
        - **Contact Information** completeness
        - **Education** details and relevance
        - **Work Experience** quality and achievements
        - **Skills** distribution and market relevance
        - **Summary/Objective** effectiveness
        
        ### 📈 What you'll get:
        - **Overall Score** with detailed breakdown
        - **Section-by-section** analysis
        - **Personalized Recommendations** for improvement
        - **Skill Analysis** with market insights
        - **Interactive Visualizations**
        
        ### 🚀 Supported Formats:
        - PDF files
        - Image files (JPG, PNG, TIFF, BMP)
        
        ---
        
        **💡 Tip**: For best results, ensure your CV is clear and well-formatted!
        """)

def display_results(results):
    """Display analysis results"""
    
    # Overall Score Section
    st.markdown("## 📊 Analysis Results")
    
    # Score cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = results['overall_score']
        color = get_score_color(overall_score)
        st.markdown(f"""
        <div class="score-card" style="background: {color};">
            <h2>{overall_score:.1f}</h2>
            <p>Overall Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        contact_score = results['section_scores']['contact_info']
        st.markdown(f"""
        <div class="metric-card">
            <h3>📞 Contact</h3>
            <h2>{contact_score:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        education_score = results['section_scores']['education']
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎓 Education</h3>
            <h2>{education_score:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        experience_score = results['section_scores']['experience']
        st.markdown(f"""
        <div class="metric-card">
            <h3>💼 Experience</h3>
            <h2>{experience_score:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Detailed Scores", "🎯 Recommendations", "🔧 Skills Analysis", "📝 Content Quality", "📄 Raw Text"])
    
    with tab1:
        display_detailed_scores(results)
    
    with tab2:
        display_recommendations(results)
    
    with tab3:
        display_skills_analysis(results)
    
    with tab4:
        display_content_quality(results)
    
    with tab5:
        display_raw_text(results)

def display_detailed_scores(results):
    """Display detailed scoring breakdown"""
    
    # Section scores chart
    section_scores = results['section_scores']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(section_scores.keys()),
            y=list(section_scores.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=[f"{v:.1f}%" for v in section_scores.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Section Scores Breakdown",
        xaxis_title="CV Sections",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Section Details")
        for section, score in section_scores.items():
            st.metric(
                label=section.replace('_', ' ').title(),
                value=f"{score:.1f}%",
                delta=get_score_delta(score)
            )
    
    with col2:
        st.subheader("📞 Contact Information")
        contact_analysis = results['contact_analysis']
        
        if 'contact_details' in contact_analysis:
            for field, present in contact_analysis['contact_details'].items():
                status = "✅" if present else "❌"
                st.write(f"{status} {field.title()}")
        
        if 'missing_fields' in contact_analysis and contact_analysis['missing_fields']:
            st.warning(f"Missing: {', '.join(contact_analysis['missing_fields'])}")

def display_recommendations(results):
    """Display improvement recommendations"""
    
    recommendations = results['recommendations']
    
    if recommendations:
        st.subheader("💡 Improvement Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 Great job! Your CV looks excellent. No major improvements needed.")
    
    # Priority recommendations
    st.subheader("🎯 Priority Actions")
    
    section_scores = results['section_scores']
    lowest_section = min(section_scores.items(), key=lambda x: x[1])
    
    st.info(f"Focus on improving your **{lowest_section[0].replace('_', ' ').title()}** section (Score: {lowest_section[1]:.1f}%)")

def display_skills_analysis(results):
    """Display skills analysis"""
    
    skill_analysis = results['skill_analysis']
    
    if not skill_analysis:
        st.warning("No skills analysis available.")
        return
    
    # Skills distribution chart
    categories = list(skill_analysis.keys())
    counts = [skill_analysis[cat]['count'] for cat in categories]
    percentages = [skill_analysis[cat]['percentage'] for cat in categories]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Skills Count by Category', 'Skills Coverage %'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=counts, name="Count", marker_color='#1f77b4'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=percentages, name="Coverage %", marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Skills details
    st.subheader("🔧 Skills Breakdown")
    
    for category, data in skill_analysis.items():
        if data['skills']:
            st.markdown(f"""
            <div class="skill-category">
                <strong>{category.title()}</strong> ({data['count']} skills, {data['percentage']:.1f}% coverage)
                <br>
                <small>{', '.join(data['skills'])}</small>
            </div>
            """, unsafe_allow_html=True)

def display_content_quality(results):
    """Display content quality analysis"""
    
    content_analysis = results['content_analysis']
    
    if 'error' in content_analysis:
        st.error(f"Content analysis error: {content_analysis['error']}")
        return
    
    # Quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📝 Word Count", content_analysis.get('word_count', 0))
        st.metric("📄 Sentences", content_analysis.get('sentence_count', 0))
    
    with col2:
        st.metric("📏 Avg Sentence Length", f"{content_analysis.get('avg_sentence_length', 0):.1f}")
        st.metric("📚 Vocabulary Diversity", f"{content_analysis.get('vocabulary_diversity', 0):.3f}")
    
    with col3:
        sentiment = content_analysis.get('sentiment', 0)
        sentiment_emoji = "😊" if sentiment > 0 else "😐" if sentiment == 0 else "😔"
        st.metric("😊 Sentiment", f"{sentiment:.3f}", delta=sentiment_emoji)
        
        readability = content_analysis.get('readability_score', 0)
        st.metric("📖 Readability", f"{readability:.1f}")

def display_raw_text(results):
    """Display extracted raw text"""
    
    st.subheader("📄 Extracted Text")
    
    # Text sections
    sections = results.get('sections', {})
    
    for section_name, section_text in sections.items():
        if section_text.strip():
            with st.expander(f"📋 {section_name.title()}"):
                st.text_area(
                    f"{section_name.title()} Content",
                    value=section_text,
                    height=150,
                    key=f"section_{section_name}"
                )
    
    # Full text
    with st.expander("📄 Full Extracted Text"):
        full_text = results.get('cleaned_text', '')
        st.text_area(
            "Complete CV Text",
            value=full_text,
            height=300,
            key="full_text"
        )

def get_score_color(score):
    """Get color based on score"""
    if score >= 90:
        return "linear-gradient(90deg, #00c851 0%, #007e33 100%)"  # Green
    elif score >= 80:
        return "linear-gradient(90deg, #33b5e5 0%, #0099cc 100%)"  # Blue
    elif score >= 70:
        return "linear-gradient(90deg, #ffbb33 0%, #ff8800 100%)"  # Orange
    else:
        return "linear-gradient(90deg, #ff4444 0%, #cc0000 100%)"  # Red

def get_score_delta(score):
    """Get delta indicator for score"""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Average"
    else:
        return "Needs Improvement"

if __name__ == "__main__":
    main()
