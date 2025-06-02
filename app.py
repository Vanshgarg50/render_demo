from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load only the model (no scaler)
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = np.array([data])
    prediction = model.predict(final_features)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)
