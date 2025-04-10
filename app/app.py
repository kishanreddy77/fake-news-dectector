from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import json

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBILwyQyqmP1exEMnoCuc-DFLsjmMvZvWI"
genai.configure(api_key=GEMINI_API_KEY)

# List available models
print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model: {m.name}")

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test_api():
    try:
        # Sample recent news article (as of March 2024)
        test_article = """
        NASA's Artemis Program: NASA has announced plans to send astronauts back to the Moon by 2026 
        as part of the Artemis program. The mission aims to establish a sustainable human presence on 
        the lunar surface and prepare for future Mars exploration. This announcement was made during 
        a press conference at NASA headquarters, with participation from international space agencies.
        """
        
        # Create the prompt for news classification
        prompt = f"""Analyze the following news article and determine if it's real or fake. 
        Consider factors like source credibility, logical consistency, and factual accuracy.
        Return your response in JSON format with 'prediction' (REAL/FAKE) and 'confidence' (0-100) fields.
        Also include a 'reasoning' field explaining your analysis.
        
        Article: {test_article}"""
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Parse the response
        try:
            response_text = response.text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                result = {
                    'prediction': 'UNKNOWN',
                    'confidence': 0,
                    'reasoning': 'Could not parse JSON response'
                }
        except Exception as e:
            result = {
                'prediction': 'UNKNOWN',
                'confidence': 0,
                'error': str(e),
                'raw_response': response_text
            }
        
        return jsonify({
            'test_article': test_article,
            'api_response': result,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Create the prompt for news classification
        prompt = f"""Analyze the following news article and determine if it's real or fake. 
        Consider factors like source credibility, logical consistency, and factual accuracy.
        Return your response in JSON format with 'prediction' (REAL/FAKE) and 'confidence' (0-100) fields.
        Also include a 'reasoning' field explaining your analysis.
        
        Article: {text}"""
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Parse the response
        try:
            response_text = response.text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                result = {
                    'prediction': 'UNKNOWN',
                    'confidence': 0,
                    'reasoning': 'Could not parse JSON response'
                }
        except Exception as e:
            result = {
                'prediction': 'UNKNOWN',
                'confidence': 0,
                'error': str(e),
                'raw_response': response_text
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)