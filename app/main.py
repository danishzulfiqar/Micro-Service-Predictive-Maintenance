import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta, timezone
import shap
from pymongo import MongoClient
from bson import ObjectId
from reportlab.lib.styles import getSampleStyleSheet
from pytz import timezone as pytz_timezone


# ----------------------------- Model Configs -----------------------------------

app = FastAPI()

# Load the model
model = load_model('./Models/Drill-14223-chenab_model.h5')

# Load the scaler
scaler = joblib.load('./Models/Drill-14223-chenab_scaler.pkl')

column_names = ['CT1', 'CT2', 'CT3', 'CT_Avg',
                'total_current', 'therm_temp', 'vibration']


# ----------------------------- MongoDB Configuration -----------------------------
# MongoDB connection

styles = getSampleStyleSheet()

# MongoDB configuration
MONGO_DB_NAME = "admin"
MONGO_URL = f"mongodb://wiser:wiser123@115.186.183.129:50001,100.28.143.179:27018,100.28.143.179:27019/admin?replicaSet=wiserReplica"

TIMEZONE_STR = "Asia/Karachi"

chenab_mac_address = "08:D1:F9:A7:4D:FC"

# Connect to MongoDB


def connect_mongo():
    client = MongoClient(MONGO_URL)
    return client[MONGO_DB_NAME]

# Fetch lab name from MongoDB

# ----------------------------- API Schema -----------------------------

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
    fault_cause: List[str]
    faulty_predicted: int
    todays_date: str
    scheduled_maintenance_date: str
    predicted_maintenance_date: str
    degraded_life: float
    falutToActiveRatio: float


# ----------------------------- Helper Functions -----------------------------

# Fetch data for a specified date range from MongoDB

def fetch_data_for_date_range(collection, query):
    entries_data = collection.find(query)
    entries_df = pd.DataFrame(list(entries_data))
    return entries_df


def convert_time_string_to_datetime(time_string):
    if not time_string:
        return None

    if isinstance(time_string, datetime):
        return time_string

    date = datetime.today().date()

    try:
        formatted_time = datetime.strptime(time_string, "%H:%M")
        combined_datetime = datetime.combine(date, formatted_time.time())
        result_string = combined_datetime.strftime("%Y-%m-%d %H:%M:%S")
        return result_string
    except ValueError as e:
        print(f"Error converting time string to datetime: {e}")
        return None


def fetch_utilization_data(mac, minutes):

    # Connect to MongoDB
    db = connect_mongo()

    # Calculate the time 15 minutes ago
    time_threshold = datetime.utcnow() - timedelta(minutes=minutes)

    # ------- Current Data -----------

    current_query = {
        "mac": mac,
        "state": "ON",
        'created_at': {'$gte': time_threshold},
    }

    lab_current_collection = db['cts']

    current_data = fetch_data_for_date_range(
        lab_current_collection, current_query)

    print("Current Data:")
    print(current_data)

    if current_data.empty:
        print("No data found")
        # return msgType:missing and message:No On state data for last 15 minutes
        return {
            "msgType": "missing",
            "message": "No On state CTS data for given minutes"
        }

    # Using only CT1, CT2, CT3, CT_Avg, total_current, state, created_at
    if not current_data.empty:
        current_data = current_data[[
            "CT1", "CT2", "CT3", "CT_Avg", "total_current", "state", "created_at"]]

    # ---------- Vibration Data ------------

    vibration_query = {
        "mac": mac,
        "created_at": {"$gte": time_threshold}
    }

    lab_vibration_collection = db['vibrations']

    vibration_data = fetch_data_for_date_range(
        lab_vibration_collection, vibration_query)

    print(vibration_data)

    if vibration_data.empty:
        print("No data found")
        return {
            "msgType": "missing",
            "message": "No vibration data for given minutes"
        }

    # Using only vibration, created_at
    if not vibration_data.empty:
        vibration_data = vibration_data[["vibration", "created_at"]]

    # ------------- Temperature Data ---------------

    temperature_query = {
        "mac": mac,
        "created_at": {"$gte": time_threshold}
    }

    lab_temperature_collection = db['thermisters']

    temperature_data = fetch_data_for_date_range(
        lab_temperature_collection, temperature_query)

    print(temperature_data)

    if temperature_data.empty:
        print("No data found")
        return {
            "msgType": "missing",
            "message": "No temperature data for given minutes"
        }

    # Using only therm_temp, created_at
    if not temperature_data.empty:
        temperature_data = temperature_data[["therm_temp", "created_at"]]

    # Joining all dataframes into one based on created_at

    if not current_data.empty and not vibration_data.empty and not temperature_data.empty:
        data = current_data.merge(vibration_data, on="created_at", how="outer").merge(
            temperature_data, on="created_at", how="outer")
    elif not current_data.empty and not vibration_data.empty:
        data = current_data.merge(vibration_data, on="created_at", how="outer")
    elif not current_data.empty and not temperature_data.empty:
        data = current_data.merge(
            temperature_data, on="created_at", how="outer")
    elif not vibration_data.empty and not temperature_data.empty:
        data = vibration_data.merge(
            temperature_data, on="created_at", how="outer")
    elif not current_data.empty:
        data = current_data
    elif not vibration_data.empty:
        data = vibration_data
    elif not temperature_data.empty:
        data = temperature_data
    else:
        data = pd.DataFrame()  # Initialize an empty DataFrame
        print("No data found")

    data = data.dropna(axis=0)

    # Generating csv file with machine name
    nodeData = db['nodes'].find_one({"mac": mac}, projection={"_id": 1})
    node_id = nodeData.get("_id")
    machine_data = db['machines'].find_one(
        {"nodeId": ObjectId(node_id)}, projection={"machineName": 1, "_id": 0})
    machine_name = machine_data.get("machineName")

    file_name = f"{machine_name}.csv"
    data.to_csv(file_name, index=False)

    # Convert data to json format
    data = data.to_dict(orient='records')

    return {
        "msgType": "success",
        "message": "Data saved successfully",
        "data": data,
        "file_name": file_name
    }


# Prediction function
def predict_data(input_data: pd.DataFrame, model, scaler, column_names, maintenance_date: str):
    # Scale the input data
    input_data[column_names] = scaler.transform(input_data[column_names])

    # Make predictions
    pred = model.predict(input_data)

    # Convert the predictions to binary
    pred = [1 if y >= 0.434683 else 0 for y in pred]


    predicted = pred

    # Shap explainer
    explainer = shap.Explainer(model, input_data)
    shap_values = explainer(input_data)

    # Find the feature that caused each anomaly
    if isinstance(shap_values, list):
        shap_df = pd.DataFrame(np.abs(shap_values[1]), columns=column_names)
    else:
        shap_df = pd.DataFrame(
            np.abs(shap_values.values), columns=column_names)

    shap_df['feature_cause'] = shap_df.idxmax(axis=1)
    fault_cause = shap_df['feature_cause'].tolist()

    # Number of faulty predictions
    faulty_predicted = pred.count(1)

    # Get today's date
    todays_date = datetime.now().strftime("%x")

    # Parse the provided maintenance date
    maintenance_date = datetime.strptime(maintenance_date, "%m/%d/%y").date()

    # Calculate predicted maintenance date
    pred_array = np.array(pred)
    n = (pred_array == 1).sum()
    days_to_subtract = n / 24 + 1
    predicted_maintenance_date = maintenance_date - \
        timedelta(days=days_to_subtract)
    predicted_maintenance_date_str = predicted_maintenance_date.strftime("%x")

    # Calculate degraded life of the machine
    degraded_life = (24 - n) / 2400

    # Calculate fault to active ratio ie number of 1's / number of input
    falutToActiveRatio = faulty_predicted / len(pred)

    return {
        "predicted": predicted,
        "fault_cause": fault_cause,
        "faulty_predicted": faulty_predicted,
        "todays_date": todays_date,
        # Convert to string
        "scheduled_maintenance_date": maintenance_date.strftime("%x"),
        "predicted_maintenance_date": predicted_maintenance_date_str,
        "degraded_life": degraded_life,
        "falutToActiveRatio": falutToActiveRatio
    }

# ----------------------------- API Endpoints -----------------------------


# Home
@app.get("/")
def read_root():
    return {"message": "Welcome to the Machine Failure Prediction API"}


# predict point for sending input json data and get the prediction
@app.post("/predict_chenab", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([item.dict() for item in request.data])

        # Call helper function to make prediction and get response
        response_data = predict_data(
            input_data, model, scaler, column_names, request.maintenance_date)

        return PredictionResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# get last minutes data to csv file
@app.get("/{macAddress}/{minutes}/get_interval_data")
def get_interval_data(macAddress: str, minutes: int):
    try:
        returnData = fetch_utilization_data(macAddress, minutes)
        if returnData.get("msgType") == "missing":
            return {"message": returnData.get("message")}
        return {"message": f"Data: {returnData}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Run prediction on the machine on given minutes data
@app.get("/{macAddress}/{minutes}/predict_machine_for_minutes")
def predict_machine_for_minutes(macAddress: str, minutes: int):
    try:
        returnData = fetch_utilization_data(macAddress, minutes)
        if returnData.get("msgType") == "missing":
            return {"message": returnData.get("message")}

        data = returnData.get("data")
        input_data = pd.DataFrame(data)

        # Using only necessary columns
        # CT1: float
        # CT2: float
        # CT3: float
        # CT_Avg: float
        # total_current: float
        # therm_temp: float
        # vibration: float

        input_data = input_data[[
            "CT1", "CT2", "CT3", "CT_Avg", "total_current", "therm_temp", "vibration"]]
        
        # csv
        input_data.to_csv("chenab_input_data.csv", index=False)

        # Call helper function to make prediction and get response

        response_data = predict_data(
            input_data, model, scaler, column_names, "01/01/21")

        return {"message": f"Data: {response_data}"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# command

# fastapi run ./app.py --port 8000

# docker

# docker build -t cenab-model .
# docker run -d --name chenab-model -p 8000:8000 chenab-model
