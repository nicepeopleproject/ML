from flask import Flask
from flask import request
from flask import jsonify
import pickle


input_file = 'model1.bin'
with open(input_file, 'rb') as f_in:
    model = pickle.load(f_in)
input_file = 'dv.bin'
with open(input_file, 'rb') as f_in:
    dv = pickle.load(f_in)
app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_prediction = model.predict_proba(X)[0, 1]
    churn = y_prediction >= 0.5
    result = {'churn_probability': y_prediction,
              'churn': bool(churn)} # numpy bool to python bool
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
