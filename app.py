from flask import Flask , render_template , request , app, jsonify , url_for
import pickle
import numpy as np

## create app

app =Flask(__name__)

## Load the model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename , 'rb'))

@app.route('/')
def home():
	return render_template('index.html')


# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=np.array(list(data.values())).reshape(1,-1)
#     output=classifier.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
