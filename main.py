from numpy import character
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

TOKEN = os.getenv('TOKEN')

app = Flask(__name__)

CORS(app, origins=["http://localhost:3000"])

@app.route('/search')
def search():
    query = request.args.get('query')
    if query is not None:
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={TOKEN}"
        r = requests.get(url)
        data = r.json()
        return jsonify({"data": data})
    else:
        return jsonify({"data": {"Error": "No query provided"}})

@app.route('/predict')
def predict():
    symbol = request.args.get('symbol')
    if symbol is not None:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={TOKEN}&outputsize=compact"
        r = requests.get(url)
        data = r.json()
        try:
            return jsonify({"data": {"Error": data['Error Message']}})
        except:
            data = data['Time Series (Daily)']
            dates = list(data.keys())
            print(dates)
            prices = []
            actual = {}
            for date in dates:
                prices.append(data[date]['4. close'])
                actual[date] = data[date]['4. close']
            prices.reverse()
            prices = np.array(prices)
            dates = np.array(list(range(len(prices))))
            dates = np.reshape(prices, (len(dates), 1))
            prices = np.reshape(prices, (len(prices), 1))
            regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            regressor.fit(dates, prices)
            prediction = regressor.predict(np.array([len(prices)]).reshape(1, -1))
            return jsonify({"data": {"Prediction": prediction.tolist(), "actual": actual}})
    else:
        return jsonify({"data": {"Error": "No symbol provided"}})

@app.route('/')
def index():
  print(request.args)
  return 'Hello, World!'


if __name__ == '__main__':
  app.run()