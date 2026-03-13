import os
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='Titanic Survival API')

class Passenger(BaseModel):
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    Sex: str
    Embarked: str
    Pclass: int

# Load the model
model = joblib.load('titanic_model.joblib')

@app.get('/')
def index():
    return {'message': 'Titanic Survival Prediction API is running'}

@app.post('/predict')
def predict(passenger: Passenger):
    # .model_dump() is preferred in Pydantic v2
    data = pd.DataFrame([passenger.model_dump()])
    
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    
    return {
        'prediction': 'Survived' if prediction == 1 else 'Did not survive',
        'survival_probability': round(probability, 4)
    }

# CRITICAL FOR RENDER DEPLOYMENT
if __name__ == "__main__":
    # Render provides the port in the environment variable $PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
