from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, date, timedelta

app = FastAPI()

# Load the model
model = load_model('./Models/chenab_model.h5')

# Load the scaler
scaler = joblib.load('./Models/chenab_scaler.pkl')

column_names = ['CT1', 'CT2', 'CT3', 'CT_Avg',
                'total_current', 'therm_temp', 'vibration']

# Define the input data model


class DataRow(BaseModel):
    CT1: float
    CT2: float
    CT3: float
    CT_Avg: float
    total_current: float
    therm_temp: float
    vibration: float


class PredictionRequest(BaseModel):
    data: List[DataRow]
    maintenance_date: str  # Date in MM/DD/YY format

# Define the output data model


class PredictionResponse(BaseModel):
    predicted: List[int]
    faulty_predicted: int
    todays_date: str
    scheduled_maintenance_date: str
    predicted_maintenance_date: str
    degraded_life: float


@app.get("/")
def read_root():
    return {"message": "Welcome to the Machine Failure Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([item.dict() for item in request.data])
        input_data[column_names] = scaler.transform(input_data[column_names])

        # Make predictions
        pred = model.predict(input_data)
        pred = [1 if y >= 0.5 else 0 for y in pred]

        predicted = pred

        # Number of faulty ie 1's predicted by the model
        faulty_predicted = pred.count(1)

        # Get today's date
        todays_date = datetime.now().strftime("%x")

        # Parse the provided maintenance date
        maintenance_date = datetime.strptime(
            request.maintenance_date, "%m/%d/%y").date()

        # Calculate predicted maintenance date
        pred_array = np.array(pred)
        n = (pred_array == 1).sum()
        days_to_subtract = n / 24 + 1
        predicted_maintenance_date = maintenance_date - \
            timedelta(days=days_to_subtract)
        predicted_maintenance_date_str = predicted_maintenance_date.strftime(
            "%x")

        # Calculate degraded life of the machine
        degraded_life = (24 - n) / 2400

        return PredictionResponse(
            predicted=predicted,
            faulty_predicted=faulty_predicted,
            todays_date=todays_date,
            scheduled_maintenance_date=request.maintenance_date,
            predicted_maintenance_date=predicted_maintenance_date_str,
            degraded_life=degraded_life
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


    

# command

# fastapi run ./app.py --port 8000

# docker

# docker build -t cenab-model .
# docker run -d --name chenab-model -p 8000:8000 chenab-model

