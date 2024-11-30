import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
from keras.models import load_model
import pickle
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utilizer import resulter


# Initialize Flask app
app = Flask(__name__)



# Flask routes
@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        data = request.get_json()
        clothing_id = data.get('clothing_id')
        print(type(clothing_id))
        if not clothing_id:
            return jsonify({"error": "No clothing_id provided"}), 400

        overall_sentiment, sentiment_counts = resulter(int(clothing_id))

        # Return JSON response with sentiment and counts
        return jsonify({
            "clothing_id": clothing_id,
            "predicted_sentiment": overall_sentiment,
            "sentiment_counts": sentiment_counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get the port from environment or default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)
# a, b = resulter(250)
# print(a)
# print(b)
 