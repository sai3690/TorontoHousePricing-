import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load the model 

RandomForestmodel = pickle.load(open('random_forest_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'})

    data = data['data']
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = RandomForestmodel.predict(new_data)
    return jsonify(output[0])







if __name__=="__main__":
    app.run(debug=True)

def __getitem__(self, key):
    return self.__dict__[key]
   