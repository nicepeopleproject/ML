import requests

host = "0.0.0.0:9696"
url = "http://%s/predict" % host
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}
print(requests.post(url, json=customer).json())