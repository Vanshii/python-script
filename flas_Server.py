from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)  # Reshape for a single prediction
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})  # Return the predicted class

if __name__ == '__main__':
    app.run(port=5000)
