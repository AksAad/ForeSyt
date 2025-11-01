from Notebooks.Train_Models import *
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/train/<symbol>')
def train(symbol):
    model = prophet_predict(symbol)
    return f"Model for {symbol} trained successfully!"

@app.route('/predict/<symbol><periods>')
def predict(symbol,periods):
    model = prophet_predict(symbol)
    future = model.make_future_dataframe(periods=periods)
    future["ds"] = pd.to_datetime(future["ds"]).dt.tz_localize(None)
    forecast = model.predict(future)
    result = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    result = result.to_dict(orient="records")

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
