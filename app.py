from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'car_name_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    # Ensure the columns match the training data
    columns = model.feature_names_in_
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[columns]
    
    prediction = model.predict(df)
    return jsonify({'predicted_car_name': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
