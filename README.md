# 📧 Spam Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)

A machine learning project that performs **spam detection** by comparing three different text representation techniques: **Bag of Words (BoW)**, **TF-IDF**, and **Word2Vec**, and evaluating their impact on classification performance.

---

## 📌 Project Overview

This project focuses on detecting spam messages using **supervised machine learning** and **natural language processing (NLP)** techniques. Three different text vectorization approaches are implemented and compared:

- Bag of Words (BoW)  
- TF-IDF  
- Word2Vec  

Each method is evaluated using accuracy and detailed classification metrics to understand their strengths and limitations in spam detection.

---

## 📁 Project Structure

- spam_detection.ipynb — Main project notebook  
- SMSSpamCollection — Dataset used for training and testing  
- bow_report.png — Accuracy and classification report using BoW  
- tfidf_report.png — Accuracy and classification report using TF-IDF  
- word2vec_report.png — Accuracy and classification report using Word2Vec  
- comparison_summary.png — Performance comparison of all methods  
- README.md — Project documentation  

---

## ⚙️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  
- Natural Language Processing (NLP)  

---

## 🧠 Models & Text Representations

### Bag of Words (BoW)
- Model: Multinomial Naive Bayes  
- Accuracy: 0.9865  
- Performs very well due to word frequency patterns  

### TF-IDF
- Model: Multinomial Naive Bayes  
- Accuracy: 0.9865  
- Slightly better weighting of important words  

### Word2Vec
- Model: Logistic Regression  
- Accuracy: 0.9040  
- Captures semantic meaning but underperforms with limited data  

---

## 📊 Performance Comparison

| Embedding | Model | Accuracy | Precision | Recall | F1-Score |
|----------|------|----------|----------|--------|----------|
| BoW | MultinomialNB | 0.9865 | 0.9677 | 0.9375 | 0.9524 |
| TF-IDF | MultinomialNB | 0.9865 | 0.9739 | 0.9313 | 0.9521 |
| Word2Vec | Logistic Regression | 0.9040 | 0.6137 | 0.8938 | 0.7277 |

---

## ▶️ How to Run

1. Clone the repository  
```text
git clone https://github.com/btboilerplate/Spam-detection.git  
```

2. Install required libraries  
```text
pip install numpy pandas matplotlib scikit-learn  
```

3. Open spam_detection.ipynb and run all cells sequentially  

---

## 🧪 Key Observations

- BoW and TF-IDF achieve the highest accuracy for spam detection  
- TF-IDF provides better word importance weighting  
- Word2Vec requires more data and tuning to perform well  
- Traditional NLP methods can outperform embeddings on small datasets  

---

## 🚀 Future Improvements

- Use pre-trained Word2Vec or GloVe embeddings  
- Try deep learning models (LSTM, CNN)  
- Apply hyperparameter tuning  
- Add cross-validation for robustness  

---
