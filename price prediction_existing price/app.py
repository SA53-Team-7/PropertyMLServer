from flask import Flask
from flask import request
from joblib import load
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def exist_property_pred():

    # get headers and convert to integer
    data = request.headers
    pid = int(data.get('projectId'))
    district = int(data.get('district'))
    area = int(data.get('floor_area'))
    floor = int(data.get('floor_range'))
    top = int(data.get('top'))
    tenure = int(data.get('tenure'))
    year = int(data.get('year'))
    month = int(data.get('month'))
    sale_type = 3
    
    model = load('resale_model.pkl')
    preds = model.predict(np.array([[pid,district,area,floor,top,tenure,year,month,sale_type]]))
    preds_as_str = str(preds)
    print(preds_as_str)
    return preds_as_str

@app.route('/', methods=['POST'])
def exist_property_pred_post():

    # get headers and convert to integer
    r = request.json
    data = r[0]
    pid = int(data['projectId'])
    district = int(data['district'])
    area = int(data['floor_area'])
    floor = int(data['floor_range'])
    top = int(data['top'])
    tenure = int(data['tenure'])
    year = int(data['year'])
    month = int(data['month'])
    sale_type = 3
    
    model = load('resale_model.pkl')
    preds = model.predict(np.array([[pid,district,area,floor,top,tenure,year,month,sale_type]]))
    preds_as_str = str(preds)
    print(preds_as_str)
    return preds_as_str

if __name__=="__main__":
    app.run()