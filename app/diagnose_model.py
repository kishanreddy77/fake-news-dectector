import joblib
import os
import numpy as np

def diagnose_model():
    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    
    # Load the model data
    print("Loading model and vectorizer...")
    model_data = joblib.load(os.path.join(models_dir, 'news_classifier.pkl'))
    
    # Extract components
    model = model_data['classifier']
    vectorizer = model_data['vectorizer']
    preprocess_text = model_data['preprocess_text']
    extract_credibility_features = model_data['extract_credibility_features']
    
    # Define test cases
    test_cases = [
        {
            'text': "NASA scientists have discovered evidence of microbial life on Mars, according to a new study published in Nature. The research team, led by Dr. Sarah Johnson, found organic compounds in soil samples collected by the Perseverance rover.",
            'expected': 1  # REAL
        },
        {
            'text': "BREAKING: SHOCKING DISCOVERY! Scientists find MIRACLE CURE for all diseases! Doctors HATE this one weird trick! Click to learn more!",
            'expected': 0  # FAKE
        },
        {
            'text': "The Federal Reserve announced a 0.25% interest rate hike today, citing persistent inflation concerns. Market analysts predict this move will impact mortgage rates and consumer spending.",
            'expected': 1  # REAL
        },
        {
            'text': "ALERT: The government is HIDING the truth about COVID vaccines! They don't want you to know about these SHOCKING side effects! Share this before they delete it!",
            'expected': 0  # FAKE
        }
    ]
    
    print("\nTesting model with sample cases...")
    print("-" * 80)
    
    for i, case in enumerate(test_cases, 1):
        # Preprocess text
        processed_text = preprocess_text(case['text'])
        
        # Extract features
        text_vector = vectorizer.transform([processed_text])
        credibility_features = extract_credibility_features(case['text'])
        
        # Combine features
        features = np.hstack((text_vector.toarray(), [credibility_features]))
        
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Print results
        print(f"\nTest Case {i}:")
        print(f"Text: {case['text'][:100]}...")
        print(f"Expected: {'REAL' if case['expected'] == 1 else 'FAKE'}")
        print(f"Predicted: {'REAL' if prediction == 1 else 'FAKE'}")
        print("\nCredibility Features:")
        print(f"- Quotes: {credibility_features[0]}")
        print(f"- Numbers/Statistics: {credibility_features[1]}")
        print(f"- Credible Source Mentions: {credibility_features[2]}")
        print(f"- Sensational Language Count: {credibility_features[3]}")
        print(f"\nCorrect: {'Yes' if prediction == case['expected'] else 'No'}")
        print("-" * 80)

if __name__ == "__main__":
    diagnose_model() 