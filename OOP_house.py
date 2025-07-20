from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pickle
import pandas as pd
import category_encoders as ce


many_labels = ['job', 'education', 'month', 'day_of_week'] #target_encoder
less_labels = ['marital', 'default', 'housing', 'loan', 'contact', 'outcome'] #one-hot

total_cols = ['age', 'job', 'education', 'month', 'day_of_week', 'duration',
       'campaign', 'days', 'previous', 'marital_divorced', 'marital_married',
       'marital_single', 'marital_unknown', 'default_no', 'default_unknown',
       'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',
       'loan_unknown', 'loan_yes', 'contact_cellular', 'contact_telephone',
       'outcome_failure', 'outcome_nonexistent', 'outcome_success']

app = FastAPI()

# Load the machine learning model and scaler
with open('model_RF.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('target_encoder.pkl', 'rb') as encoder_file:
    target_encoder = pickle.load(encoder_file)

class InputData(BaseModel):
    age: int
    job: object
    marital: object
    education: object
    default: object
    housing: object
    loan: object
    contact: object
    month: object
    day_of_week: object
    duration: float
    campaign: int
    days: int
    previous: int
    outcome: object

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post('/predict')
def predict(input_data: InputData):
    data = input_data.dict()
    data_df = pd.DataFrame([data])

    data_df["days"] = data_df["days"].replace(999, 0)

    data_df = target_encoder.transform(data_df)
    data_df = pd.get_dummies(data_df, columns=less_labels)
    data_df.iloc[:, 9:] = data_df.iloc[:, 9:].astype(int)

    data_df = data_df.reindex(columns=total_cols)

    prediction = model.predict(data_df)
    prediction = 'yes' if int(prediction) == 1 else 'no'
    return {"prediction": prediction}

