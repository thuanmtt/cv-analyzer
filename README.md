# CV Analyzer - AI-Powered Resume Evaluation System

## Mô tả
Hệ thống đánh giá CV sử dụng AI để phân tích và đánh giá CV từ file PDF hoặc ảnh. Hệ thống có khả năng:
- Trích xuất thông tin từ CV (PDF/ảnh)
- Phân tích và đánh giá chất lượng CV
- Dự đoán điểm số và đề xuất cải thiện
- Giao diện web thân thiện với Streamlit

## Tính năng chính
- 📄 **Xử lý đa định dạng**: PDF, JPG, PNG
- 🤖 **AI Analysis**: Sử dụng model transformer để phân tích nội dung
- 📊 **Visualization**: Biểu đồ và thống kê chi tiết
- 🎯 **Scoring**: Hệ thống chấm điểm tự động
- 💡 **Recommendations**: Đề xuất cải thiện CV

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd cv-analyzer
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Cài đặt Tesseract OCR (cho xử lý ảnh)
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Windows
# Tải từ: https://github.com/UB-Mannheim/tesseract/wiki
```

### 4. Tải model spaCy
```bash
python -m spacy download en_core_web_sm
```

## Sử dụng

### Chạy ứng dụng Streamlit
```bash
streamlit run app.py
```

### Huấn luyện model
```bash
python train_model.py
```

### Tải dataset từ Kaggle
```bash
python download_dataset.py
```

## Cấu trúc project
```
cv-analyzer/
├── app.py                 # Streamlit app chính
├── models/               # Thư mục chứa model
├── data/                 # Dataset và dữ liệu
├── utils/                # Utility functions
├── training/             # Scripts huấn luyện
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Dataset
Project sử dụng dataset từ Kaggle:
- **Resume Dataset**: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- **Job Posting Dataset**: https://www.kaggle.com/datasets/promptcloud/job-posting-dataset

## Model Architecture
- **Text Extraction**: PyPDF2 + Tesseract OCR
- **Text Processing**: spaCy + NLTK
- **Classification**: Transformer-based model (BERT)
- **Scoring**: Custom scoring algorithm

## License
MIT License
