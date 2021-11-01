import pickle

input_file = 'model1.bin'
with open(input_file, 'rb') as f_in:
    model = pickle.load(f_in)
input_file = 'dv.bin'
with open(input_file, 'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
X = dv.transform([customer])
y_prediction = model.predict_proba(X)[0, 1]
print(y_prediction)