import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from tensorflow import keras

# Load model
model = keras.models.load_model('iris_model')

# Load input scaling parameters
with open('scaling.json') as f:
    d = json.load(f)

app = FastAPI()


@app.get("/predict")
async def get_predition(f0: float, f1: float, f2: float, f3: float):
    f0 = (f0 - d['means'][0]) / np.sqrt(d['vars'][0])
    f1 = (f1 - d['means'][1]) / np.sqrt(d['vars'][1])
    f2 = (f2 - d['means'][2]) / np.sqrt(d['vars'][2])
    f3 = (f3 - d['means'][3]) / np.sqrt(d['vars'][3])
    X_scaled = [[f0, f1, f2, f3]]
    
    y_pred = model.predict(X_scaled)
    df_pred = pd.DataFrame({
        'Species': ['Virginica', 'Versicolor', 'Setosa'],
        'Confidence': y_pred.flatten()
    })
    df_pred['Confidence'] = [round(x,4) for x in df_pred['Confidence']]
    df_pred.set_index('Species', inplace=True)
    return df_pred.to_dict()
