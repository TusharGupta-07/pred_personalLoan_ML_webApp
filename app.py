import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('random_forest_reg.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #model.predict_proba(test)[:,1]


    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted value of customer is : {}'.format(output))


if __name__ == '__main__':
    app.debug = True
    app.run()
