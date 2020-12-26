import numpy as np
from flask import Flask, request,  render_template
import pickle

from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(degree = 2)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.form
    int_features = []
    int_features.append(int(data['age']))
    int_features.append(float(data['bmi']))
    int_features.append(int(data['children']))
    int_features.append(int(data['smoker']))
    
    final_features = [np.array(int_features)]
    
    # Polynomial transformation before applying the model and then applying the prediction model
    prediction = model.predict(pol.fit_transform(final_features))
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted cost value of health insurance will be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)