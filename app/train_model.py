import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import re

def preprocess_text(text):
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?$%()-]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_credibility_features(text):
    # Count quotes (potential citations)
    quotes = len(re.findall(r'"([^"]*)"', text))
    
    # Count numbers (potential statistics)
    numbers = len(re.findall(r'\d+(?:\.\d+)?%?', text))
    
    # Check for credible source mentions
    credible_sources = ['study', 'research', 'professor', 'dr.', 'university', 'scientist', 'expert', 'report', 'analysis']
    source_count = sum(1 for source in credible_sources if source in text.lower())
    
    # Check for sensational language
    sensational_words = ['shocking', 'incredible', 'amazing', 'unbelievable', 'miracle', 'secret', 'breakthrough']
    sensational_count = sum(1 for word in sensational_words if word in text.lower())
    
    return [quotes, numbers, source_count, sensational_count]

def train_model():
    # Read the data
    print("Loading dataset...")
    df = pd.read_csv('data/news.csv')
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Extract credibility features
    print("Extracting credibility features...")
    credibility_features = np.array([extract_credibility_features(text) for text in df['text']])
    
    # Get the labels
    labels = df['label']
    
    # Split the dataset
    print("Splitting dataset...")
    x_text_train, x_text_test, x_cred_train, x_cred_test, y_train, y_test = train_test_split(
        df['processed_text'], credibility_features, labels, 
        test_size=0.2, random_state=7, stratify=labels
    )
    
    # Initialize TfidfVectorizer with improved parameters
    print("Vectorizing text data...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000
    )
    
    # Fit and transform text features
    tfidf_train = tfidf_vectorizer.fit_transform(x_text_train)
    tfidf_test = tfidf_vectorizer.transform(x_text_test)
    
    # Combine TF-IDF features with credibility features
    train_features = np.hstack((tfidf_train.toarray(), x_cred_train))
    test_features = np.hstack((tfidf_test.toarray(), x_cred_test))
    
    # Initialize PassiveAggressiveClassifier with balanced weights
    print("Training model...")
    pac = PassiveAggressiveClassifier(
        max_iter=100,
        C=0.5,
        class_weight='balanced',
        random_state=42
    )
    pac.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred = pac.predict(test_features)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {score*100:.2f}%')
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
    
    # Print confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model, vectorizer, and preprocessing functions
    print("\nSaving model and vectorizer...")
    model_data = {
        'classifier': pac,
        'vectorizer': tfidf_vectorizer,
        'preprocess_text': preprocess_text,
        'extract_credibility_features': extract_credibility_features
    }
    joblib.dump(model_data, 'models/news_classifier.pkl')
    print("Training completed successfully!")

if __name__ == "__main__":
    train_model() 