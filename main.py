from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Bankruptcy Prediction API",
    description="Predicts whether a company will go bankrupt",
    version="1.0"
)

# Load trained pipeline model
with open("bankruptcy_model.pkl", "rb") as file:
    model = pickle.load(file)


# Define input data structure
class CompanyData(BaseModel):
    industrial_risk: float
    management_risk: float
    financial_flexibility: float
    credibility: float
    competitiveness: float
    operating_risk: float


# Home route
@app.get("/")
def home():
    return {"message": "Bankruptcy Prediction API is running"}


# Prediction route
@app.post("/predict")
def predict(data: CompanyData):

    input_data = np.array([[
        data.industrial_risk,
        data.management_risk,
        data.financial_flexibility,
        data.credibility,
        data.competitiveness,
        data.operating_risk
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "bankruptcy" if prediction == 1 else "non-bankruptcy"

    return {
        "prediction": result,
        "probability_of_bankruptcy": round(float(probability)*100, 2)
    }