from copyreg import pickle
from flask import Flask,request,jsonify,app,url_for,render_template
import pickle
import pandas as pd
import numpy as np
import warnings


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def homepage():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict_ans():
    data = [float(i) for i in request.form.values()]
    final_features = [np.array(data)]
    output = model.predict(final_features)
    print(output)

    return render_template('home.html',prediction_text="Airfoil pressure is  {} decibles".format(output[0]))


if __name__=="__main__":
    app.run(debug=True)