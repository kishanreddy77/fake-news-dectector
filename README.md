# News Detector AI

A powerful web application that uses Google's Gemini AI to detect and analyze fake news articles in real-time.

## Features

- **Real-time Analysis**: Get instant results on news authenticity
- **Smart Detection**: Uses Gemini AI for advanced content analysis
- **Detailed Explanations**: Provides context-specific explanations for why content is classified as real or fake
- **User-friendly Interface**: Modern, responsive design with smooth animations
- **Multiple Categories**: Specialized analysis for different types of news (political, scientific, celebrity, etc.)

## Categories of Analysis

The application can detect and provide specific explanations for various types of fake news:

- **Weather/Climate News**: Analyzes claims about weather manipulation and climate change
- **Health/Vaccine News**: Evaluates medical and vaccine-related information
- **Extraterrestrial News**: Assesses claims about aliens and UFOs
- **Conspiracy Theories**: Identifies unsubstantiated conspiracy claims
- **Celebrity News**: Detects sensational and fabricated celebrity stories
- **Political News**: Analyzes political claims and government-related information

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd news-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
   - Get your API key from [Google's MakerSuite](https://makersuite.google.com/app/apikey)
   - Add your API key to `app/app.py`

4. Run the application:
```bash
python app/app.py
```

## Usage

1. Open your browser and navigate to `http://localhost:5003`
2. Paste or type your news article in the text area
3. Click "Analyze News" to get the results
4. View the detailed analysis and explanation

## Response Format

The application provides:
- **Prediction**: REAL NEWS or FAKE NEWS
- **Detailed Explanation**: Context-specific analysis of why the content is classified as real or fake
- **Visual Indicators**: Color-coded results (green for real, red for fake)

## Requirements

- Python 3.7+
- Flask
- google-generativeai
- A valid Gemini API key

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 