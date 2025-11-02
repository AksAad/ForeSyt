import base64
import io
import os
from Notebooks.Train_Models import prophet_predict
from flask import Flask, jsonify
from flask import send_from_directory
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
app = Flask(__name__)

