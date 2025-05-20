# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY :- CODTECH IT SOLUTIONS

NAME :- SOUMODEEP BISWAS

INTERN ID :- CT04DM1074

DOMAIN :- PYTHON PROGRAMMING

DURATION :- 4 WEEKS

DURATION :- NEELA SANTOSH

# **Spam Detection System Using Machine Learning: Description**

This Python script implements a **text classification system** that distinguishes between spam and legitimate messages (ham) using machine learning techniques. The implementation demonstrates a complete pipeline from data preparation to model evaluation, with a focus on email/spam message classification.

## **1. Overview**
The system uses **Natural Language Processing (NLP)** and **supervised machine learning** to classify text messages as either **spam (1)** or **ham (0)**. Key components include:
- **Text vectorization** using TF-IDF (Term Frequency-Inverse Document Frequency)
- **Naive Bayes classifier** (MultinomialNB) for classification
- **Model evaluation** with accuracy, classification report, and confusion matrix
- **Visualization** of results using Matplotlib and Seaborn

---

## **2. Key Features**
### **2.1. Data Preparation**
- A **sample dataset** is created manually for demonstration, containing:
  - **Spam messages** (e.g., "Get free money now!!!")
  - **Legitimate messages** (e.g., "Hi John, how about a meeting tomorrow?")
- In a real-world scenario, this would be replaced with a larger dataset (e.g., SMS Spam Collection, Enron Spam Dataset).

### **2.2. Text Preprocessing & Feature Extraction**
- **TF-IDF Vectorizer** converts text into numerical features:
  - Removes **stop words** (common words like "the," "is") to reduce noise.
  - Limits features to the **top 1000 words** (`max_features=1000`).
  - Computes **TF-IDF scores**, which weigh words based on importance.

### **2.3. Model Training & Evaluation**
- **Train-Test Split:** The dataset is divided into **70% training** and **30% testing**.
- **Multinomial Naive Bayes (MNB):** A probabilistic classifier well-suited for text classification.
- **Performance Metrics:**
  - **Accuracy score** (percentage of correct predictions)
  - **Classification report** (precision, recall, F1-score)
  - **Confusion matrix** (visualized using Seaborn)

### **2.4. Prediction on New Data**
- The trained model is used to classify **new, unseen messages** (e.g., "Free viagra!!!").
- Each prediction is labeled as **Spam (1)** or **Ham (0)**.

---

## **3. Technical Implementation**
### **3.1. Libraries Used**
- **Pandas & NumPy:** Data handling
- **Scikit-learn (sklearn):** 
  - `TfidfVectorizer` (text vectorization)
  - `MultinomialNB` (Naive Bayes classifier)
  - `train_test_split` (data splitting)
  - `accuracy_score`, `classification_report`, `confusion_matrix` (evaluation)
- **Matplotlib & Seaborn:** Visualization

### **3.2. Workflow**
1. **Data Loading:** A small dataset is created for demonstration.
2. **Feature Extraction:** Text is converted into TF-IDF vectors.
3. **Model Training:** The Naive Bayes classifier learns from the training data.
4. **Evaluation:** Performance is measured on the test set.
5. **Visualization:** Confusion matrix is plotted.
6. **Prediction:** New messages are classified.

### **3.3. Limitations**
- **Small Dataset:** The example uses only 8 messages; real-world models need thousands.
- **Basic Preprocessing:** No lemmatization, stemming, or advanced NLP techniques.
- **No Cross-Validation:** A single train-test split may not reflect true performance.

---

## **4. Example Output**
### **4.1. Model Evaluation**
```
Accuracy: 1.0  

Classification Report:  
              precision  recall  f1-score  support  
           0       1.00    1.00      1.00         2  
           1       1.00    1.00      1.00         1  
    accuracy                           1.00         3  
   macro avg       1.00    1.00      1.00         3  
weighted avg       1.00    1.00      1.00         3  
```
### **4.2. Confusion Matrix**
A heatmap showing:
- **True Negatives (TN):** Correctly predicted ham.
- **True Positives (TP):** Correctly predicted spam.
- (No false positives/negatives in this small example.)

### **4.3. New Email Predictions**
```
Email: 'Free viagra!!!' - Spam  
Email: 'Hello, please find attached the report' - Ham  
Email: 'Congratulations! You've won a free ticket' - Spam  
```

---

## **5. Potential Enhancements**
1. **Larger Dataset:** Use real-world datasets (e.g., SMS Spam Collection).
2. **Advanced NLP:** Apply **lemmatization, n-grams, or word embeddings** (Word2Vec).
3. **Better Models:** Try **Logistic Regression, Random Forest, or Neural Networks**.
4. **Hyperparameter Tuning:** Optimize TF-IDF and model parameters.
5. **Deployment:** Convert into a **Flask/Django API** for real-time classification.

---

## **6. Applications**
- **Email Spam Filtering**
- **SMS Fraud Detection**
- **Social Media Moderation**
- **Automated Customer Support**

---

## **7. Conclusion**
This script provides a **foundation for text classification** using machine learning. While simplified, it demonstrates key concepts in **NLP, feature extraction, and model evaluation**. With enhancements, it can be adapted for production-level spam detection systems.
