"""
Train CV analysis model using transformer-based architecture
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CVDataset(Dataset):
    """Custom dataset for CV data"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class CVClassifier(nn.Module):
    """Transformer-based CV classifier"""
    
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3):
        super(CVClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class CVModelTrainer:
    """Trainer class for CV analysis model"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Prepare training data"""
        # Load sample data
        if os.path.exists('data/sample/sample_cvs.json'):
            with open('data/sample/sample_cvs.json', 'r') as f:
                data = json.load(f)
        else:
            # Create synthetic data for training
            data = self._create_synthetic_data()
        
        texts = []
        labels = []
        
        for cv in data:
            # Combine all CV sections
            text = f"{cv.get('summary', '')} {cv.get('experience', '')} {cv.get('skills', '')}"
            texts.append(text)
            
            # Create quality labels based on score
            score = cv.get('score', 70)
            if score >= 90:
                label = 'excellent'
            elif score >= 80:
                label = 'good'
            elif score >= 70:
                label = 'average'
            else:
                label = 'poor'
            
            labels.append(label)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        return texts, encoded_labels
    
    def _create_synthetic_data(self) -> List[Dict]:
        """Create synthetic CV data for training"""
        synthetic_data = []
        
        # Excellent CVs
        for i in range(20):
            cv = {
                'summary': f'Experienced professional with {5+i} years of expertise in software development. Passionate about creating innovative solutions and leading technical teams.',
                'experience': f'Senior Software Engineer at TechCompany{i} (2020-2023)\n- Led development of scalable microservices\n- Managed team of {5+i} developers\n- Implemented CI/CD pipelines\nSoftware Developer at Startup{i} (2018-2020)\n- Developed REST APIs using Python and Django\n- Optimized database queries improving performance by 40%',
                'skills': 'Python, JavaScript, React, Django, PostgreSQL, AWS, Docker, Kubernetes, Git, Jenkins, Jira',
                'score': 90 + i
            }
            synthetic_data.append(cv)
        
        # Good CVs
        for i in range(30):
            cv = {
                'summary': f'Skilled developer with {3+i} years of experience in web development. Dedicated to writing clean, maintainable code.',
                'experience': f'Software Developer at DevCompany{i} (2020-2023)\n- Developed web applications using React and Node.js\n- Collaborated with cross-functional teams\nJunior Developer at SmallStartup{i} (2018-2020)\n- Built responsive user interfaces\n- Participated in code reviews',
                'skills': 'JavaScript, React, Node.js, HTML, CSS, Git, MongoDB',
                'score': 80 + i
            }
            synthetic_data.append(cv)
        
        # Average CVs
        for i in range(25):
            cv = {
                'summary': f'Developer with {2+i} years of experience. Looking for opportunities to grow and learn new technologies.',
                'experience': f'Junior Developer at Company{i} (2021-2023)\n- Worked on frontend development\n- Fixed bugs and implemented features\nIntern at Startup{i} (2020-2021)\n- Assisted with basic development tasks',
                'skills': 'HTML, CSS, JavaScript, Python, Git',
                'score': 70 + i
            }
            synthetic_data.append(cv)
        
        # Poor CVs
        for i in range(15):
            cv = {
                'summary': f'Recent graduate seeking entry-level position. Eager to learn and contribute to team projects.',
                'experience': f'Student at University{i}\n- Completed coursework in computer science\n- Participated in hackathons',
                'skills': 'Basic programming, Microsoft Office',
                'score': 50 + i
            }
            synthetic_data.append(cv)
        
        return synthetic_data
    
    def train(self, texts: List[str], labels: List[int], 
              batch_size: int = 8, epochs: int = 5, learning_rate: float = 2e-5):
        """Train the model"""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = CVDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = CVDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        num_classes = len(self.label_encoder.classes_)
        model = CVClassifier(self.model_name, num_classes)
        model.to(self.device)
        
        # Setup training
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(model, 'models/best_cv_model.pth')
        
        # Save label encoder
        self.save_label_encoder()
        
        # Print final results
        print(f'\nBest validation accuracy: {best_val_acc:.2f}%')
        
        # Classification report
        print('\nClassification Report:')
        print(classification_report(all_labels, all_predictions, 
                                  target_names=self.label_encoder.classes_))
        
        return model
    
    def save_model(self, model, path: str):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f'Model saved to {path}')
    
    def save_label_encoder(self):
        """Save label encoder"""
        os.makedirs('models', exist_ok=True)
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print('Label encoder saved to models/label_encoder.pkl')
    
    def load_model(self, model_path: str) -> CVClassifier:
        """Load trained model"""
        num_classes = len(self.label_encoder.classes_)
        model = CVClassifier(self.model_name, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model


def main():
    """Main training function"""
    print("CV Analyzer - Model Training")
    print("=" * 40)
    
    # Initialize trainer
    trainer = CVModelTrainer()
    
    # Prepare data
    print("Preparing training data...")
    texts, labels = trainer.prepare_data('data/sample/sample_cvs.json')
    print(f"Training data prepared: {len(texts)} samples")
    
    # Train model
    print("\nStarting model training...")
    model = trainer.train(texts, labels, batch_size=4, epochs=3)
    
    print("\nTraining completed successfully!")
    print("Model files saved in 'models/' directory")


if __name__ == "__main__":
    main()
