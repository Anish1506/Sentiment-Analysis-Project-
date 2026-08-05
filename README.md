# Twitter Sentiment Analysis using Ensemble of Machine Learning and Deep Learning Models

An ensemble-based Twitter Sentiment Analysis system that combines **BERT+CNN+BiLSTM**, **GloVe+CNN+BiLSTM**, and **Support Vector Machine (SVM)** to classify tweets into **Positive**, **Neutral**, and **Negative** sentiments. The ensemble model leverages the strengths of both traditional machine learning and deep learning to achieve improved classification performance.

## 📌 Overview

Social media platforms generate massive amounts of textual data that reflect public opinions. This project analyzes COVID-19-related Twitter posts using an ensemble learning approach to improve sentiment classification accuracy.

The proposed system integrates predictions from three different models using an ensemble voting strategy to produce robust sentiment predictions.

## 🚀 Features

- Text preprocessing and cleaning
- BERT-based contextual embeddings
- GloVe word embeddings
- CNN + BiLSTM hybrid architecture
- SVM classifier with Count Vectorization
- Ensemble prediction using voting
- Interactive Streamlit GUI
- Real-time sentiment prediction
- Confidence score visualization

## 🏗️ Model Architecture

### Model 1
- BERT
- CNN
- BiLSTM

### Model 2
- GloVe Embeddings
- CNN
- BiLSTM

### Model 3
- Support Vector Machine (SVM)
- Count Vectorizer

### Ensemble

The outputs of all three models are combined using a voting mechanism to generate the final sentiment prediction.

## 🛠️ Tech Stack

- Python
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- cuML (GPU SVM)
- Pandas
- NumPy
- NLTK
- Streamlit
- Matplotlib

## 📂 Dataset

The COVID-19 Twitter Dataset contains comprising 143,903 COVID-19 related tweets. For experimentation, 115,123 tweets were used for training, and 28,780 were reserved for testing the proposed models. tweets categorized into:

- Positive
- Neutral
- Negative

## ⚙️ Preprocessing

The following preprocessing steps are applied:

- Lowercase conversion
- HTML tag removal
- Punctuation removal
- Number removal
- Stopword removal
- Multiple space removal
- Single character removal
- Tokenization

## 📈 Results

| Model | Accuracy |
|--------|----------|
| BERT + CNN + BiLSTM | **92.55%** |
| GloVe + CNN + BiLSTM | **94.22%** |
| SVM | **87.23%** |
| **Ensemble Model** | **94.95%** |

The ensemble model achieved the highest accuracy by combining predictions from all three models.

## 💻 GUI

The project includes a Streamlit-based graphical interface where users can:

- Enter custom text
- Predict sentiment
- View confidence scores
- Analyze sample tweets

## 📊 Future Improvements

- Multilingual sentiment analysis
- Explainable AI (XAI)
- Dynamic ensemble weighting
- Real-time Twitter streaming
- Model optimization
- Cross-domain adaptation
- Lightweight deployment for edge devices
