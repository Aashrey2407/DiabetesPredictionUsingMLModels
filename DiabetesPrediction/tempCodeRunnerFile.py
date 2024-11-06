import pickle
from flask import Flask,request,jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)

CORS(app)

with open('naive_bayes_model.pkl','rb') as file:
    naive_bayes_model = pickle.load(file)
with open('perceptron_model.pkl','rb') as file:
    perceptron_model = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
@app.route('/predict',methods = ['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model_type')
    input_features = np.array([[data['glucose'],data['insulin'],data['bmi'],data['age']]])
    #Choose the model based on input
    input_features = scaler.transform(input_features)
    if model_type == 'naive_bayes':
        prediction = naive_bayes_model.predict(input_features)
    elif model_type == 'perceptron':
        prediction = perceptron_model.predict(input_features)
    else:
        return jsonify({'error':'Invalid model type'}),400

    return jsonify({'diabetes_type':int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)