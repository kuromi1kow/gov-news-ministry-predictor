 Ministry Classification Project
This repository contains a full pipeline for classifying Kazakhstani news articles into relevant ministries using Natural Language Processing (NLP) and machine learning. The system consists of two specialized models and a main multi-class neural network model.

📦 Dataset
The dataset includes manually collected and labeled news articles in Russian. Each sample contains:

title – headline of the article

content – main body of the article

ministry – the manually labeled ministry

text – combined field used for classification (title + content)

🧩 Models Breakdown
✅ 1. General Ministry Classifier (Neural Network)
A deep learning model built with TensorFlow/Keras that classifies articles into one of four major ministries:

Министерство энергетики (Ministry of Energy)

Министерство внутренних дел (Ministry of Internal Affairs)

Министерство экологии и природных ресурсов (Ministry of Ecology & Natural Resources)

Министерство сельского хозяйства (Ministry of Agriculture)

Model Highlights:

Combines keyword vector input + embedded token sequence

Uses TextVectorization, Embedding, Conv1D, and GlobalMaxPooling

Evaluated with accuracy/loss plots

Saved to file as ministry_classifier_model.keras

🎯 2. Trade Ministry Classifier (Traditional ML)
A separate model focused specifically on detecting news for:

Министерство торговли и интеграции (Ministry of Trade and Integration)

Model Details:

Lemmatization and tokenization using Natasha

Keyword-based feature extraction via custom MintradeKeywordCounter

Tested with models like XGBoost, MultinomialNB, LogisticRegression

Best result achieved with: XGBoostClassifier (accuracy: ~0.938)

🧪 3. Model Training and Evaluation
Data is split using train_test_split (with stratification)

Classifiers are compared using:

accuracy_score

classification_report (precision, recall, f1-score)

Final predictions tested on new, unseen articles
Input:
![image](https://github.com/user-attachments/assets/cda52626-5070-44ce-a359-9c6b837f3ac3)
Output:
![image](https://github.com/user-attachments/assets/1dc9bb6e-d80a-46a5-a4e1-c219cee1a014)

📊 Visualization
Model structure visualized using plot_model()

Training performance shown with matplotlib:

Accuracy per epoch

Loss per epoch

🚀 Technologies Used
Python, Pandas, NumPy

TensorFlow/Keras

Natasha (for Russian NLP)

NLTK (for stopword removal and tokenizing)

XGBoost, Scikit-learn
