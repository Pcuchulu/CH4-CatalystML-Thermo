from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import StringIO  # Correct import for StringIO

app = Flask(__name__)


model_name = "bert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def process_prediction(input_data):
    
    prediction = np.random.rand(3) * 100  
    mse = mean_squared_error([90, 80, 70], prediction)  
    r2 = r2_score([90, 80, 70], prediction)
    return prediction, mse, r2

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    
    prediction, mse, r2 = process_prediction(input_data)

    
    return jsonify({
        'prediction': prediction.tolist(),
        'mse': mse,
        'r2': r2
    })

@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    data = request.get_json()
    csv_data = data['csv_data']
    
    
    csv_buffer = StringIO(csv_data)
    df = pd.read_csv(csv_buffer)
    
    predictions = []
    mse_list = []
    r2_list = []

    for _, row in df.iterrows():
        input_data = row.to_dict()
        prediction, mse, r2 = process_prediction(input_data)
        predictions.append(prediction.tolist())  
        mse_list.append(mse)  
        r2_list.append(r2)    

    return jsonify({
        'prediction': predictions,
        'mse': mse_list,
        'r2': r2_list
    })

if __name__ == '__main__':
    app.run(debug=True)
