# Twitter Sentiment Analysis using an Ensemble of BERT, CNN-BiLSTM, and SVM

This project predicts the sentiment (**Positive** / **Negative**) of tweets using an ensemble (voting-based) architecture that combines three independent classification pipelines: a **BERT-based transformer channel**, a **CNN-BiLSTM deep learning channel**, and a **classical SVM channel**. The final prediction is produced through a **voting mechanism** that aggregates the outputs of all three channels.

## Overview

Raw tweets are first cleaned through a common preprocessing step, after which the pipeline branches into three parallel channels. Each channel independently learns to classify sentiment using a different representation and modeling strategy. Their individual predictions are combined at a voting stage to produce the final sentiment label.

## Architecture

The diagram below summarizes the end-to-end pipeline:

```
Tweets for Review
      │
      ▼
 Preprocessing
      │
   ┌──┴────────────────┬─────────────────────┐
   ▼                   ▼                     ▼
Channel 1           Channel 2             Channel 3
(BERT)          (CNN + BiLSTM)              (SVM)
   │                   │                     │
   ▼                   ▼                     ▼
BERT               Tokenizer            Count Vectorizer
Preprocessor            │                     │
   │                 Padding                  ▼
   ▼                   │                     SVM
Embedding          Vectorization               │
   │                   │                       │
Transformer         GloVe Embedding             │
Encoder                 │                       │
   │              ┌─────┴─────┐                 │
Classification +   ▼           ▼                │
GELU + Norm       CNN        BiLSTM             │
   │           (Conv Layer  (stacked            │
Final Embedding  + Maxpool)   LSTM layers)       │
Outcome              │           │               │
   │                 └─────┬─────┘               │
   └──────────────────┐    │    ┌────────────────┘
                       ▼    ▼    ▼
                        Voting
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
             Positive           Negative
```

### 1. Preprocessing
Raw tweets are cleaned (e.g., noise removal, normalization) before being fed into the three channels.

### 2. Channel 1 — BERT
- **BERT Preprocessor** tokenizes and formats text for BERT.
- **Embedding** layer generates contextual token embeddings.
- **Transformer Encoder** captures contextual/semantic relationships across the sequence.
- **Classification + GELU + Norm** applies a classification head with GELU activation and normalization.
- **Final Embedding Outcome** is passed forward as the output of the BERT channel, and also feeds into the CNN-BiLSTM channel.

### 3. Channel 2 — CNN + BiLSTM (hybrid deep learning channel)
- **Tokenizer** converts text into token sequences.
- **Padding** standardizes sequence lengths.
- **Vectorization** converts tokens into numerical form.
- **GloVe Embedding** maps tokens to pre-trained GloVe word vectors.
- The embedded sequence (combined with the BERT channel's output) is passed to:
  - **CNN** — a convolutional layer followed by max-pooling to extract local n-gram / feature patterns from the embeddings.
  - **BiLSTM** — stacked bidirectional LSTM layers that model sequential/contextual dependencies in both directions.
- The CNN and BiLSTM outputs feed into the voting stage.

### 4. Channel 3 — SVM (classical ML channel)
- **Count Vectorizer** converts tweets into a bag-of-words / term-frequency representation.
- **SVM (Support Vector Machine)** classifies the vectorized text.

### 5. Voting (Ensemble)
Predictions from the BERT channel, the CNN-BiLSTM channel, and the SVM channel are combined using a **voting** strategy to produce the final sentiment label: **Positive** or **Negative**.

## Why an Ensemble?

- **BERT** captures deep contextual and semantic meaning from language.
- **CNN-BiLSTM** captures local n-gram patterns (via CNN) and long-range sequential dependencies in both directions (via BiLSTM).
- **SVM** provides a strong, lightweight classical baseline using sparse text representations.

Combining these diverse models through voting helps reduce individual model bias/variance and generally improves robustness and accuracy over any single model.

## Tech Stack

| Component | Tool / Library |
|---|---|
| Transformer model | BERT (Hugging Face Transformers) |
| Word embeddings | GloVe |
| Deep learning | CNN, BiLSTM (TensorFlow / Keras or PyTorch) |
| Classical ML | SVM (scikit-learn) |
| Text vectorization | Tokenizer, Count Vectorizer |
| Language | Python |

## Project Structure

```
├── data/                   # Raw and processed tweet data
├── preprocessing/          # Text cleaning and preprocessing scripts
├── channel1_bert/          # BERT preprocessing, embedding, and classification
├── channel2_cnn_bilstm/    # Tokenizer, padding, GloVe embedding, CNN, BiLSTM
├── channel3_svm/           # Count vectorizer and SVM model
├── ensemble/                # Voting logic combining all channel predictions
├── notebooks/               # Experiments and evaluation notebooks
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Usage
```bash
# 1. Preprocess raw tweets
python preprocessing/preprocess.py --input data/raw_tweets.csv --output data/clean_tweets.csv

# 2. Train / run each channel
python channel1_bert/run_bert.py
python channel2_cnn_bilstm/run_cnn_bilstm.py
python channel3_svm/run_svm.py

# 3. Combine predictions via voting
python ensemble/vote.py --output predictions.csv
```

## Results

*(Add your accuracy, precision, recall, F1-score, and confusion matrix here once available.)*

| Model | Accuracy | F1-Score |
|---|---|---|
| BERT | — | — |
| CNN-BiLSTM | — | — |
| SVM | — | — |
| **Ensemble (Voting)** | — | — |

## Future Work

- Experiment with weighted voting instead of majority voting.
- Add more sentiment classes (e.g., Neutral).
- Deploy as a REST API for real-time tweet sentiment scoring.

## License

Add your preferred license here (e.g., MIT).
