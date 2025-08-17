# 🚀 CV Analyzer - Quick Start Guide

## 📋 Tổng quan
CV Analyzer là hệ thống đánh giá CV sử dụng AI với giao diện Streamlit. Hệ thống có thể phân tích CV từ file PDF hoặc ảnh và đưa ra đánh giá chi tiết.

## ⚡ Khởi động nhanh

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Tạo sample data
```bash
python create_sample_data.py
```

### 3. Chạy ứng dụng
```bash
streamlit run app.py
```

### 4. Truy cập ứng dụng
Mở trình duyệt và truy cập: `http://localhost:8501`

## 🎯 Tính năng chính

### 📄 Xử lý đa định dạng
- **PDF files**: Sử dụng PyPDF2 và pdfplumber
- **Image files**: JPG, PNG, TIFF, BMP với OCR (Tesseract)

### 🤖 AI Analysis
- **Text Extraction**: Trích xuất text từ CV
- **Section Analysis**: Phân tích từng phần của CV
- **Scoring System**: Hệ thống chấm điểm tự động
- **Recommendations**: Đề xuất cải thiện

### 📊 Visualization
- **Interactive Charts**: Biểu đồ tương tác với Plotly
- **Score Breakdown**: Phân tích chi tiết điểm số
- **Skill Analysis**: Phân tích kỹ năng theo danh mục

## 🔧 Các script hữu ích

### Demo và Testing
```bash
# Chạy demo để test các tính năng
python demo.py

# Xử lý dataset từ archive
python process_archive_data.py
```

### Training Model
```bash
# Huấn luyện model (cần dataset)
python training/train_model.py
```

## 📁 Cấu trúc project

```
cv-analyzer/
├── app.py                 # Ứng dụng Streamlit chính
├── demo.py               # Script demo
├── create_sample_data.py # Tạo sample data
├── process_archive_data.py # Xử lý dataset
├── utils/                # Utility functions
│   ├── text_extractor.py # Trích xuất text
│   └── cv_analyzer.py    # Phân tích CV
├── training/             # Scripts huấn luyện
├── data/                 # Dataset và dữ liệu
├── models/               # Model đã train
└── archive/              # Dataset từ Kaggle
```

## 🎨 Giao diện

### Sidebar
- **Upload CV**: Tải lên file CV
- **Features**: Mô tả tính năng
- **About**: Thông tin về ứng dụng

### Main Content
- **Overall Score**: Điểm tổng quan
- **Section Scores**: Điểm từng phần
- **Recommendations**: Đề xuất cải thiện
- **Skills Analysis**: Phân tích kỹ năng
- **Content Quality**: Chất lượng nội dung

## 📈 Scoring System

### Các tiêu chí đánh giá:
1. **Contact Information** (10%): Thông tin liên hệ
2. **Education** (20%): Học vấn
3. **Experience** (30%): Kinh nghiệm làm việc
4. **Skills** (25%): Kỹ năng
5. **Summary** (15%): Tóm tắt

### Thang điểm:
- **90-100**: Excellent
- **80-89**: Good
- **70-79**: Average
- **<70**: Needs Improvement

## 🔍 Skill Categories

Hệ thống phân loại kỹ năng thành các nhóm:
- **Programming**: Python, Java, JavaScript, etc.
- **Databases**: MySQL, PostgreSQL, MongoDB, etc.
- **Frameworks**: Django, React, Angular, etc.
- **Cloud**: AWS, Azure, Docker, Kubernetes, etc.
- **Tools**: Git, Jenkins, Jira, etc.
- **Languages**: English, Spanish, etc.

## 🚨 Troubleshooting

### Lỗi thường gặp:

1. **ModuleNotFoundError**: Cài đặt lại dependencies
```bash
pip install -r requirements.txt
```

2. **spaCy model not found**: Tải model
```bash
python -m spacy download en_core_web_sm
```

3. **Tesseract not found**: Cài đặt Tesseract OCR
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr
```

4. **Port already in use**: Thay đổi port
```bash
streamlit run app.py --server.port 8502
```

## 📝 Ví dụ sử dụng

### 1. Upload CV
- Chọn file PDF hoặc ảnh từ sidebar
- Click "Analyze CV"

### 2. Xem kết quả
- Overall Score: Điểm tổng quan
- Section Scores: Điểm từng phần
- Recommendations: Đề xuất cải thiện

### 3. Phân tích chi tiết
- Skills Analysis: Phân tích kỹ năng
- Content Quality: Chất lượng nội dung
- Raw Text: Text đã trích xuất

## 🎯 Next Steps

1. **Upload CV thật** để test
2. **Xử lý dataset** từ archive
3. **Huấn luyện model** với dữ liệu thực
4. **Tùy chỉnh scoring** theo nhu cầu
5. **Deploy** lên cloud platform

## 📞 Support

Nếu gặp vấn đề, hãy kiểm tra:
1. Dependencies đã cài đặt đầy đủ
2. Model spaCy đã tải
3. Tesseract OCR đã cài đặt
4. File CV có định dạng hỗ trợ
