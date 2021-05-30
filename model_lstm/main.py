import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from flask import Flask, request

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)

# Model Load
model = tf.keras.models.load_model('base_lstm')
# Encoder Load
data_enc = LabelEncoder()
data_enc.classes_ = np.load("data_classes.npy", allow_pickle=True)

label_enc = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
       'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
       'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
       'Common Cold', 'Dengue', 'Diabetes ',
       'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
       'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
       'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
       'Osteoarthristis', 'Paralysis (brain hemorrhage)',
       'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
       'Typhoid', 'Urinary tract infection', 'Varicose veins',
       'hepatitis A']

# Data Disease Desc and caution
data_disesase = pd.read_csv("data_disease_precautions.csv")

def predict_model(data_arr):
    data_pre = data_enc.transform(np.reshape(data_arr, [17, 1]))
    data_pre = np.reshape(data_pre, [1, 1, 17])
    predict_result = model.predict(data_pre)
    ind = np.argmax(predict_result)
    if  (np.sort(predict_result)[-1][-1]) > 0.999:
        return 1, label_enc[ind], np.sort(predict_result)[-1][-1]
    else:
        return 0, label_enc[ind], np.sort(predict_result)[-1][-1]

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json
    print("data: {}".format(request_json))
    print("type: {}".format(type(request_json)))

    data = request_json.get('data')
    resp, predict, prob = predict_model(data)
    if resp != 0 :
        loc = data_disesase.loc[data_disesase["Disease"]==predict]
        des = loc.Description.values
        precaution = list(loc.iloc[0, 1:5].values)
        response_json = {
            "resp" : 1,
            "probability" : str(prob),
            "prediction" : str(predict),
            "desc" : str(des),
            "precautions" : precaution
            
        }

        return json.dumps(response_json)
    else:
        loc = data_disesase.loc[data_disesase["Disease"]==predict]
        des = loc.Description.values
        precaution = list(loc.iloc[0, 1:5].values)
        response_json = {
            "resp" : 0,
            "probability" : str(prob),
            "prediction" : str(predict),
            "desc" : str(des),
            "precautions" : precaution
            
        }
        return json.dumps(response_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)