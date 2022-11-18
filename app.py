import pickle
from flask import Flask,app,render_template,request,url_for,jsonify
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier

app=Flask(__name__)

model=pickle.load(open('xgb_credit.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/credit_api',methods=['POST'])
def card():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    input=(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(input)[0]
    print(output)
    return jsonify(int(output))



@app.route('/credit',methods=['POST'])
def credit():
    data=[x for x in request.form.values()]
    print(data)
    input=(np.array(data).reshape(1,-1))
    print(input)
    output=model.predict(input)[0]
    print(output)
    res=''
    if output==1:
        res=" Default payment"
    else:
        res="NO Default payment"
        
    return render_template('result.html',
    result=" [{}] based on credit card owner's characteristics and payment history".format(res))
    
      
if __name__ == "__main__":
    app.run(debug=True)
    