import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from tensorflow import keras
from pydantic import BaseModel

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
        
# Load model
model = keras.models.load_model('iris_model')

# Load input scaling parameters
with open('scaling.json') as f:
    s = json.load(f)

app = FastAPI()

@app.get("/predict")
async def get_predition(iris: Iris):
    f0 = (iris.sepal_length - s['means'][0]) / np.sqrt(s['vars'][0])
    f1 = (iris.sepal_width - s['means'][1]) / np.sqrt(s['vars'][1])
    f2 = (iris.petal_length - s['means'][2]) / np.sqrt(s['vars'][2])
    f3 = (iris.petal_width - s['means'][3]) / np.sqrt(s['vars'][3])
    X_scaled = [[f0, f1, f2, f3]]
    
    y_pred = model.predict(X_scaled)
    df_pred = pd.DataFrame({
        'Species': ['Virginica', 'Versicolor', 'Setosa'],
        'Confidence': y_pred.flatten()
    })
    df_pred['Confidence'] = [round(x,4) for x in df_pred['Confidence']]
    df_pred.set_index('Species', inplace=True)
    return df_pred.to_dict()