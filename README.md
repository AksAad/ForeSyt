<img width="1919" height="924" alt="image" src="https://github.com/user-attachments/assets/1c3396c0-9f3e-4122-8488-60736d331ebe" />
ForeSyt is a quick stock price predictor which displays the prices for the next 30-business days.
I made this as a passion project just to learn and improve, the model most certainly can make errors and be inaccurate, I have made sure to display the error metrics on the plot image.
<img width="1919" height="933" alt="image" src="https://github.com/user-attachments/assets/47c2c494-07b6-4f1b-8dc1-2de52aee4845" />
The fetch_data.py file contains a function which fetches the last 10 year data from the yfinance module, the train_moodels.py contains functions for 3 models (Xgboost, Random Forest Regressor and Meta Prophet).
Meta Prophet is a time-series forecast model, so it is the most accurate among the 3, which is why I make use of that in the first place.
The website was designed by my buddy @Tezzy-4202, 
Follow these steps to use the project:
  1. Git clone this repository (click on the green code box on top and copy paste the link "git clone <link>" into your terminal)
  2. pip install or uv install relevant modules
  3. run the all_endpoint.py file
  4. use the website
  

Thanks for reading
