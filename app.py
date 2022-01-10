from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model_ann = load_model('dwi_ann_model.h5')
model_cnn = load_model('dwi_cnn_model.h5')

class_dict = {0: 'Seledri', 1: 'Sirih'}

def predict_label(img_path, method):
    query = cv2.imread(img_path)
    output = query.copy()
    query = cv2.resize(query, (32, 32))
    q = []
    q.append(query)
    q = np.array(q, dtype='float') / 255.0
    if (method == "ann"):
        q_pred = model_ann.predict(q)
    elif (method == "cnn"):
        q_pred = model_cnn.predict(q)
    else:
        return 'error: invalid method name'
    if q_pred<=0.5 :
        predicted_bit = 0
    else :
        predicted_bit = 1
    return class_dict[predicted_bit]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ann', methods=['GET', 'POST'])
def ann():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path, 'ann')
            return render_template('ann.html', uploaded_image=image.filename, prediction=prediction)
    return render_template('ann.html')

@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path, 'cnn')
            return render_template('cnn.html', uploaded_image=image.filename, prediction=prediction)
    return render_template('cnn.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)