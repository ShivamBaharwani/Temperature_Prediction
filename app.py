from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting features from the form
        features = [float(x) for x in request.form.values()]
        
        # Scaling features
        scaled_features = scaler.transform([features])
        
        # Making predictions
        prediction = model.predict(scaled_features)
        
        return render_template('results.html', prediction=prediction[0])
    
if __name__ == "__main__":
    app.run(debug=True)
