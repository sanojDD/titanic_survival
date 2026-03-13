from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title='Titanic Survival API')

# Define the input data schema
class Passenger(BaseModel):
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    Sex: str
    Embarked: str
    Pclass: int

# Load the model once at startup
model = joblib.load('titanic_model.joblib')

@app.get('/')
def index():
    return {'message': 'Titanic Survival Prediction API is running'}

@app.post('/predict')
def predict(passenger: Passenger):
    # Convert Pydantic model to DataFrame
    data = pd.DataFrame([passenger.dict()])
    
    # Make prediction
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    
    return {
        'prediction': 'Survived' if prediction == 1 else 'Did not survive',
        'survival_probability': round(probability, 4)
    }