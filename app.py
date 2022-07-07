##Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


def home():
    return render_template('index.html')


def predict():
    '''
    For rendering results on HTML GUI, Write your Code here
    '''
    print(request.form)
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


# Template source:https://www.free-css.com/