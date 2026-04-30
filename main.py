from flask import Flask, request, jsonify
from flask_cors import CORS
import matlab.engine

app = Flask(__name__)
CORS(app) # Allows your HTML file to talk to this server

print("==================================================")
print("WAKING UP MATLAB ENGINE... PLEASE WAIT (10-20 seconds)")
print("==================================================")
eng = matlab.engine.start_matlab()
print("MATLAB ENGINE ONLINE. Server listening on port 5000.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Catch the data sent from the HTML webpage
        data = request.json
        
        # 2. Extract the 13 features in the exact order MATLAB expects
        inputs = [
            float(data['age']), float(data['sex']), float(data['cp']),
            float(data['trestbps']), float(data['chol']), float(data['fbs']),
            float(data['restecg']), float(data['thalach']), float(data['exang']),
            float(data['oldpeak']), float(data['slope']), float(data['ca']),
            float(data['thal'])
        ]

        # 3. Convert Python list to MATLAB double array
        matlab_input = matlab.double([inputs])

        # 4. Execute your live_predict.m script
        result = eng.live_predict(matlab_input)

        # 5. Send the percentage back to the browser
        return jsonify({'probability': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)