## Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

## Create flask app
app = Flask(__name__)

## Load random forest model
model = pickle.load(open('model.pkl', 'rb'))

## First decorator for rendering html home page

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    model_output = round(prediction[0])

    if model_output == 0:
        return render_template('index.html', prediction_text='Outcome is {}. Congratulations you are well.'.format(model_output))
    else:
        return render_template('index.html', prediction_text='Outcome is {}. Recommending further tests'.format(model_output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)