import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

try:
    with open('naive_bayes_model.pkl', 'rb') as file:
        naive_bayes_model = pickle.load(file)
    logging.info("Naive Bayes model loaded successfully")
except Exception as e:
    logging.error(f"Error loading naive_bayes_model: {e}")
    exit(1)

try:
    with open('perceptron_model.pkl', 'rb') as file:
        perceptron_model = pickle.load(file)
    logging.info("Perceptron model loaded successfully")
except Exception as e:
    logging.error(f"Error loading perceptron_model: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not all(k in data for k in ['age', 'glucose', 'insulin', 'bmi']):
            return jsonify({'error': 'Missing required input data'}), 400

        model_type = data.get('model_type', 'naive_bayes')
        input_features = np.array([[data['age'], data['glucose'], data['insulin'], data['bmi']]])

        logging.info(f"Received input: {input_features}")

        if model_type == 'naive_bayes':
            prediction = naive_bayes_model.predict(input_features)
        elif model_type == 'perceptron':
            prediction = perceptron_model.predict(input_features)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        logging.info(f"Prediction result: {prediction[0]}")
        return jsonify({'diabetes_type': int(prediction[0])})

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logging.info("Starting Flask app")
    app.run(debug=True)
