from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load('models/xgb_model.pkl')

# Define request schema
class ShopperData(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: int
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    Weekend: int
    VisitorType_New_Visitor: int
    VisitorType_Other: int

@app.post("/predict")
def predict(data: ShopperData):
    input_data = np.array([list(data.dict().values())])
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
