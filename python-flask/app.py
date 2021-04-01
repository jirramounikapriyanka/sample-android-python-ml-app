import numpy as np
from flask import Flask,request
from flask_cors import CORS
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
CORS(app)

@app.route('/',methods=['POST','GET'])
def home():
    return 'Hello, jirra here....'

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[]
    print(request)
    fixed=request.json['fixed acidity']
    int_features.append(float(fixed))
    volatile=request.json['volatile acidity']
    int_features.append(float(volatile))
    citric=request.json['citric acid']
    int_features.append(float(citric))
    residual_sugar=request.json['residual sugar']
    int_features.append(float(residual_sugar))
    chlorides=request.json['chlorides']
    int_features.append(float(chlorides))
    free_sulfur_dioxide=request.json['free sulfur dioxide']
    int_features.append(float(free_sulfur_dioxide))
    total_sulfur_dioxide=request.json['total sulfur dioxide']
    int_features.append(float(total_sulfur_dioxide))
    density=request.json['density']
    int_features.append(float(density))
    ph=request.json['pH']
    int_features.append(float(ph))
    sulphates=request.json['sulphates']
    int_features.append(float(sulphates))
    alcohol=request.json['alcohol']
    int_features.append(float(alcohol))
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=int(prediction)
    if output==0:
        return {'message':'wine is bad please donot have it'}
    else:
        return  {'message':'wine is good you can  have it'}

if __name__=="__main__":
    app.run(debug=True)
