from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# Load models and scalers
models = {
    'co2_conversion': {
        'model': joblib.load('saved_models\\co2 conversion ratio_mlp.pkl'),
        'scaler': joblib.load('saved_models\\co2 conversion ratio_scaler.pkl')
    },
    'ch4_yield': {
        'model': joblib.load('saved_models\\ch4 yield (percent)_mlp.pkl'),
        'scaler': joblib.load('saved_models\\ch4 yield (percent)_scaler.pkl')
    },
    'ch4_selectivity': {
        'model': joblib.load('saved_models\\ch4 selectivity (percent)_mlp.pkl'),
        'scaler': joblib.load('saved_models\\ch4 selectivity (percent)_scaler.pkl')
    }
}

# Input features in the correct order
FEATURE_ORDER = [
    'Active component type formation energy',
    'Active component type density',
    'Active component content (wt percent)',
    'Promoter type formation energy',
    'Promoter type density',
    'Promoter content (wt percent)',
    'Support a type formation energy',
    'Support a type density',
    'Support a content (wt percent)',
    'Support b type formation energy',
    'Support b type density',
    'Calcination Temperature (C)',
    'Calcination time (h)',
    'Reduction Temperature (C)',
    'Reduction Pressure (bar)',
    'Reduction time (h)',
    'Reduced hydrogen content (vol percent)',
    'Temperature (C)',
    'Pressure (bar)',
    'Weight hourly space velocity [mgcat/(min·ml)]',
    'Content of inert components in raw materials (vol percent)',
    'h2/co2 ratio (mol/mol)'
]


def predict(model_key, user_input):
    try:
        # Ensure input is in the correct order
        user_df = pd.DataFrame([user_input])

        # Apply scaling
        scaler = models[model_key]['scaler']
        user_scaled = scaler.transform(user_df[FEATURE_ORDER])

        # Predict
        model = models[model_key]['model']
        prediction = model.predict(user_scaled)

        return prediction[0]

    except Exception as e:
        return str(e)


@app.route('/predict/co2_conversion', methods=['POST'])
def predict_co2_conversion():
    user_input = request.json
    prediction = predict('co2_conversion', user_input)
    return jsonify({"co2_conversion_ratio": f"{prediction:.2f} %"})

@app.route('/predict/ch4_yield', methods=['POST'])
def predict_ch4_yield():
    user_input = request.json
    prediction = predict('ch4_yield', user_input)
    return jsonify({"ch4_yield": f"{prediction:.2f} %"})

@app.route('/predict/ch4_selectivity', methods=['POST'])
def predict_ch4_selectivity():
    user_input = request.json
    prediction = predict('ch4_selectivity', user_input)
    return jsonify({"ch4_selectivity": f"{prediction:.2f} %"})

CORS(app)  # Add this line after initializing the app
# Main execution
if __name__ == '__main__':
    app.run(debug=True)

