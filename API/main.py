from fastapi import FastAPI
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load Model
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('xgbboost_1.model')  # load data

#Load Encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)

@app.get("/")
async def root():
    return {print (encoder.classes_)}

# link tutorial https://towardsdatascience.com/create-your-first-rest-api-in-fastapi-e728ae649a60
# link docs https://fastapi.tiangolo.com/tutorial/
