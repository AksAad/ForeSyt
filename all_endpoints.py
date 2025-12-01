from flask import Flask, render_template, request, redirect, url_for
from Notebooks.Train_Models import prophet_predict   # uses your existing function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    ticker = request.form.get('user_input_field', '').strip()
    if not ticker:
        return redirect(url_for('home'))
    prophet_predict(ticker, periods=30)
    return redirect(url_for('show_result', ticker=ticker))
@app.route('/result/<ticker>')
def show_result(ticker):
    image_filename = f"{ticker}_prophet.png"
    image_url = url_for('static', filename=image_filename)
    return render_template(
        'result.html',
        ticker=ticker.upper(),
        image_url=image_url
    )

if __name__ == '__main__':
    app.run(debug=True)
