#!/usr/bin/env python
# coding: utf-8

# In[77]:


import import_ipynb
from Notebooks.fetch_data import get_stock_data
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import re
import matplotlib.pyplot as plt


# In[92]:


def train_xgboost_model(ticker: str):
    df = get_stock_data(ticker.lower())
    X = df.drop(['Date', 'Close'], axis=1)
    Y = df["Close"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, shuffle=False)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 1,
        "colsample_bytree": 0.8,
        "seed": 101
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    y_pred = model.predict(dvalid)
    validation_dates = df['Date'].iloc[-len(y_valid):]
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae = mean_absolute_error(y_valid, y_pred)
    mape = np.mean(np.abs((y_valid - y_pred) / (y_valid + 1e-9))) * 100

    print(f"Model Performance for {ticker.upper()}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    plt.figure(figsize=(12, 6))
    plt.plot(validation_dates, y_valid.values, label='Actual Close Price', linewidth=2)
    plt.plot(validation_dates, y_pred, label='Predicted Close Price', linewidth=2)
    plt.title(f'{ticker.upper()} Model Analysis', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel("Predicted 'Close' Price of the Stock")
    plt.legend()
    plt.grid(True)
    plt.show()
    return model


# In[93]:




# In[95]:


from sklearn.preprocessing import StandardScaler


# In[96]:


def train_random_forest(ticker: str):
    df = get_stock_data(ticker.lower())
    X = df.drop(['Date', 'Close'], axis=1)
    y = df['Close']
    split_idx = int(len(df) * 0.7)
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_valid = df['Date'].iloc[split_idx:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        oob_score=False
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_valid_scaled)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae = mean_absolute_error(y_valid, y_pred)
    mape = np.mean(np.abs((y_valid - y_pred) / (y_valid + 1e-9))) * 100

    print(f"Model Performance for {ticker.upper()}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    plt.figure(figsize=(12, 6))
    plt.plot(dates_valid, y_valid.values, label='Actual Close Price', linewidth=2)
    plt.plot(dates_valid, y_pred, label='Predicted Close Price', linewidth=2)
    plt.title(f'{ticker.upper()} Random Forest Model Analysis', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel("Predicted 'Close' Price of the Stock")
    plt.legend()
    plt.grid(True)
    plt.show()
    return model


# In[97]:




# In[98]:


def prophet_predict(ticker: str, periods: int = 30):

    original_ticker = ticker  
    ticker = ticker.strip().lower()


    df = get_stock_data(ticker)         
    df = df.sort_values('Date').reset_index(drop=True)


    df_prophet = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)

    lag_features = [col for col in df_prophet.columns if "lag" in col]


    split_idx = int(len(df_prophet) * 0.7)
    train = df_prophet.iloc[:split_idx].copy()
    validation = df_prophet.iloc[split_idx:].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    for name in lag_features:
        model.add_regressor(name)

    model.fit(train)

    if lag_features:
        forecast_val = model.predict(validation[["ds"] + lag_features])
    else:
        forecast_val = model.predict(validation[["ds"]])

    y_true = validation["y"].values
    y_pred = forecast_val["yhat"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\nModel Performance for {original_ticker.upper()}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")


    full_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    for name in lag_features:
        full_model.add_regressor(name)

    full_model.fit(df_prophet)


    periods = int(periods)  

    last_date = df_prophet["ds"].max()
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=periods
    )

    y_history = list(df_prophet["y"].values)

    future_rows = []

    def extract_lag_steps(col_name: str) -> int:
        """
        Try to infer lag length from name (e.g. 'lag_1', 'close_lag5').
        If not found, default to 1.
        """
        m = re.search(r"(\d+)", col_name)
        return int(m.group(1)) if m else 1

    for d in future_dates:
        row = {"ds": d}

        for name in lag_features:
            lag_k = extract_lag_steps(name)
            if len(y_history) >= lag_k:
                row[name] = y_history[-lag_k]
            else:
                row[name] = y_history[0] 

        row_df = pd.DataFrame([row])
        pred_row = full_model.predict(row_df)

        yhat = float(pred_row["yhat"].iloc[0])
        yhat_lower = float(pred_row["yhat_lower"].iloc[0])
        yhat_upper = float(pred_row["yhat_upper"].iloc[0])

        row["yhat"] = yhat
        row["yhat_lower"] = yhat_lower
        row["yhat_upper"] = yhat_upper

        future_rows.append(row)
        y_history.append(yhat) 

    forecast_future = pd.DataFrame(future_rows)


    fig, ax = plt.subplots(figsize=(12, 6))

    forecast_start = forecast_future["ds"].iloc[0]
    zoom_start = forecast_start - pd.Timedelta(days=60)
    zoom_end = forecast_future["ds"].iloc[-1]


    df_zoom = df.copy()
    df_zoom["Date"] = pd.to_datetime(df_zoom["Date"]).dt.tz_localize(None)
    mask_hist = (df_zoom["Date"] >= zoom_start) & (df_zoom["Date"] <= zoom_end)

    ax.plot(
        df_zoom.loc[mask_hist, "Date"],
        df_zoom.loc[mask_hist, "Close"],
        label="Actual Close",
        linewidth=1.6,
    )

    mask_val_zoom = (forecast_val["ds"] >= zoom_start) & (forecast_val["ds"] <= last_date)
    ax.plot(
        forecast_val.loc[mask_val_zoom, "ds"],
        forecast_val.loc[mask_val_zoom, "yhat"],
        label="Validation Forecast",
        linewidth=1.6,
    )
    ax.fill_between(
        forecast_val.loc[mask_val_zoom, "ds"],
        forecast_val.loc[mask_val_zoom, "yhat_lower"],
        forecast_val.loc[mask_val_zoom, "yhat_upper"],
        alpha=0.18,
        label="Validation Uncertainty",
    )


    ax.plot(
        forecast_future["ds"],
        forecast_future["yhat"],
        label=f"Next {periods}-Business-Day Forecast",
        linewidth=2.0,
    )
    ax.fill_between(
        forecast_future["ds"],
        forecast_future["yhat_lower"],
        forecast_future["yhat_upper"],
        alpha=0.3,
        label="Future Uncertainty",
    )


    ax.axvline(
        forecast_start,
        linestyle="--",
        linewidth=1.2,
        label="Forecast Start",
    )


    ax.set_xlim(zoom_start, zoom_end)


    train_start, train_end = train["ds"].min().date(), train["ds"].max().date()
    val_start, val_end = validation["ds"].min().date(), validation["ds"].max().date()

    fig.suptitle(
        f"{original_ticker.upper()} Closing Price Forecast (Next {periods} Business Days)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(
        f"Train: {train_start}–{train_end} | Val: {val_start}–{val_end}",
        fontsize=10,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Close Price")

    metrics_text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%"
    ax.text(
        0.01,
        0.99,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{original_ticker}_prophet.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return full_model, df


if __name__ == "__main__":
    train_xgboost_model("NVDA")
    train_xgboost_model("AAPL")
    train_random_forest("AAPL")
    prophet_predict("NVDA")
    prophet_predict("AAPL", 50)
