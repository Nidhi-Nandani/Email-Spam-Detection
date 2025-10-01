import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import joblib

# Load dataset
df = pd.read_csv('mail_dataset.csv')

# Replace NaNs with empty strings 
data = df.where(pd.notnull(df), '')

# Print dataset info
data.info()
print("Dataset shape:", data.shape)

# Map labels: spam -> 0, ham -> 1
data['label'] = data['label'].astype(str)
data.loc[data['label'].str.lower() == 'spam', 'label'] = 0
data.loc[data['label'].str.lower() == 'ham', 'label'] = 1
data['label'] = data['label'].astype(int)

# Check class balance
print("Label distribution before balancing:\n", data['label'].value_counts())

# Balance the dataset by oversampling the minority class
spam = data[data['label']==0]
ham = data[data['label']==1]

if len(spam) > len(ham):
    ham = resample(ham, replace=True, n_samples=len(spam), random_state=42)
elif len(ham) > len(spam):
    spam = resample(spam, replace=True, n_samples=len(ham), random_state=42)

data_balanced = pd.concat([spam, ham])
print("Label distribution after balancing:\n", data_balanced['label'].value_counts())

# Split into features and labels
X = data_balanced['text']
Y = data_balanced['label']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF vectorizer with unigrams + bigrams
feature_extraction = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Train Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_features, Y_train)

# Accuracy on training set
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training dataset:', accuracy_on_training_data)

# Accuracy on test set
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

# Print confusion matrix and classification report for test set
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, prediction_on_test_data))
print("\nClassification Report:")
print(classification_report(Y_test, prediction_on_test_data, target_names=["Spam", "Ham"]))

# Save model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(feature_extraction, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully!")
