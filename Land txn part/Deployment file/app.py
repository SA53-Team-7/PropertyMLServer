from flask import Flask
from flask import request
from joblib import load
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def new_property_pred():
    
    data = request.headers
#     return str(data);
    land_price = int(data.get("land_price"))
    
    model = load('predict_new_model.pkl')
    preds = model.predict(np.array([[land_price]]))
    preds_as_str = str(preds)
    print(preds_as_str)
    return preds_as_str

@app.route('/', methods=['POST'])
def new_property_pred_post():

    # get headers and convert to integer
    r = request.json
#     return str(r)
    data = r
    land_price = int(data['land_price'])
    
    model = load('predict_new_model.pkl')
    preds = model.predict(np.array([[land_price]]))
    preds_as_str = str(preds)
    print(preds_as_str)
    return preds_as_str

if __name__=="__main__":
    app.run()