from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle

# creating a Flask app
app = Flask(__name__)

# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/get_preds', methods = ['POST'])
def predict():
    data = request.json
    query = data["array"]
    clf = pickle.load("modelPortableDoctor.pkl")
    ans = clf.predict(query)
    return jsonify(list(ans)[0])



# driver function
if __name__ == '__main__':

    app.run()
