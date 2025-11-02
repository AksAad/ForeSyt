import base64
import io

from Notebooks.Train_Models import *
from flask import Flask, jsonify
from flask import render_template, request
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
app = Flask(__name__)
@app.route("/home")
def home():
    return render_template("index.html")

@app.route('/train/<symbol>')
def train(symbol):
    model, df = prophet_predict(symbol)
    return jsonify({"message": f"Model for {symbol.upper()} trained successfully!"})


@app.route('/predict/<symbol>/<int:periods>')
def predict(symbol, periods):
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    model, df = prophet_predict(symbol, periods=periods)
    sys.stdout = old_stdout
    logs = mystdout.getvalue()
    metrics = {}
    for line in logs.splitlines():
        if "RMSE" in line:
            metrics["RMSE"] = float(line.split(":")[1].strip())
        elif "MAE" in line:
            metrics["MAE"] = float(line.split(":")[1].strip())
        elif "MAPE" in line:
            metrics["MAPE"] = float(line.split(":")[1].replace("%","").strip())
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    interpretation = (
        f"The model predicts {periods}-day ahead prices for {symbol.upper()} with an "
        f"RMSE of {metrics.get('RMSE', 0):.2f} and MAPE of {metrics.get('MAPE', 0):.2f}%. "
        f"Lower MAPE and RMSE indicate better performance. "
        "The forecast shows future trend continuation based on historical data."
    )
    return jsonify({
        "symbol": symbol.upper(),
        "periods": periods,
        "metrics": metrics,
        "interpretation": interpretation,
        "forecast_graph": image_base64
    })


if __name__ == '__main__':
    app.run(debug=True)
