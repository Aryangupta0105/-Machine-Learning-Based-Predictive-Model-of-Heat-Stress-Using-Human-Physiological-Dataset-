from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_gb.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract all 11 features in the order they appear in the CSV training
        features = [
            float(data['Mean_Tbody']),
            float(data['LSR']),
            float(data['LDF']),
            float(data['HR_initial']),
            float(data['SBP']),
            float(data['DBP']),
            float(data['MAP']),
            float(data['SSNA']),
            float(data['Heat_exposure']),
            float(data['Sweat_rate']),
            float(data['change'])  # This is the 11th parameter
        ]
        
        final_features = np.array([features])
        tcore_pred = model.predict(final_features)[0]
        status = "Hyperthermic" if tcore_pred >= 38.0 else "Normothermic"
        
        return jsonify({
            'tcore': round(tcore_pred, 2),
            'status': status
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5005, debug=True, use_reloader=False)
    
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    app.run(host='0.0.0.0', port=port)
