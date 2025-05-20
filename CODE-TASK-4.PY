# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (using a sample spam dataset)
# In a real scenario, you would use your own dataset
# For this example, I'll use a sample dataset from sklearn
from sklearn.datasets import fetch_20newsgroups

# Since sklearn doesn't have a spam dataset, let's simulate one
# In practice, you would load your actual spam dataset here
# Let's create a simple example dataset
data = {
    'text': [
        'Get free money now!!!', 
        'Hi John, how about a meeting tomorrow?',
        'Win a million dollars today!',
        'Meeting reminder: Project discussion at 3pm',
        'Limited time offer: Buy one get one free',
        'Your account statement is ready',
        'Claim your prize now!',
        'Team lunch next Friday'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham
}

df = pd.DataFrame(data)

# Text preprocessing and feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model (Naive Bayes is good for text classification)
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Example of using the model to predict new emails
new_emails = [
    "Free viagra!!!", 
    "Hello, please find attached the report",
    "Congratulations! You've won a free ticket"
]

new_emails_transformed = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_transformed)

print("\nPredictions for new emails:")
for email, pred in zip(new_emails, predictions):
    print(f"Email: '{email}' - {'Spam' if pred == 1 else 'Ham'}")
