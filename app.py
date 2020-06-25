from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))
data_columns = json.load(open('columns.json','r'))['data_columns']
locations = data_columns[3:]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    location = request.form['location']
    total_sqft = float(request.form['area'])
    bhk = request.form['bhk']
    bath = request.form['bath']
    
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1
        
    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    output = round(model.predict([x])[0],2)
    
    return render_template('index.html', result='Predicted Price is {} lacs'.format(output))
    
if __name__ == "__main__":
    app.run(debug=True)