![Forecast Plot](https://github.com/user-attachments/assets/1c3396c0-9f3e-4122-8488-60736d331ebe)

## Overview

**ForeSyt** is a quick stock price prediction tool that forecasts prices for the next **30 business days**.
This project was built as a **passion project** to learn and improve skills in time-series forecasting, machine learning, and full-stack integration.

**Disclaimer**:
This model is **not intended for financial advice**. Predictions can be inaccurate, and errors are expected. To maintain transparency, **error metrics are displayed directly on the forecast plots**.


## Models Used

I have experimented with multiple forecasting approaches:

* **Meta Prophet** (Primary Model)

  * Time-series–specific forecasting model
  * Performs best among the three and is used for final predictions

* **XGBoost Regressor**

* **Random Forest Regressor**

Meta Prophet generally outperforms the other models due to its ability to handle trends, seasonality, and holidays in financial time series.

---

## Project Structure

* `fetch_data.py`
  Fetches **10 years of historical stock data** using the `yfinance` module.

* `train_models.py`
  Contains training logic for:

  * XGBoost
  * Random Forest Regressor
  * Meta Prophet

* `all_endpoint.py`
  Backend entry point to serve predictions to the frontend.

---

## Error Metrics & Visualization

![Error Metrics Plot](https://github.com/user-attachments/assets/47c2c494-07b6-4f1b-8dc1-2de52aee4845)

All prediction plots include **error metrics** to provide a realistic picture of model performance and limitations.

---

## Frontend
The website interface was designed by my buddy:  **@Tezzy-4202**

---

## How to Run the Project

1. **Clone the repository**

   ```bash
   git clone <repository-link>
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   or

   ```bash
   uv install
   ```

3. **Run the backend**

   ```bash
   python all_endpoint.py
   ```

4. **Open the website** and start predicting

---

## Motivation
This project was built purely for:

* Learning time-series forecasting
* Experimenting with multiple ML models
* Understanding model error and limitations
* End-to-end ML + web integration

---

## Final Notes

If you find issues, inaccuracies, or have suggestions, feel free to open an issue or submit a pull request.

Thanks for reading.
