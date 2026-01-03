from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, computed_field, Field, field_validator
from typing import Literal, Annotated
import pickle 
import pandas as pd
from schema.user_input import UserInput

#import the ML model 
with open('model/insurance_premium_model.pkl', 'rb') as f:
    model = pickle.load(f)

#ML flow
MODEL_VERSION = '1.0.0'

app = FastAPI()




@app.get('/')
def home():
    return {'message': 'Insurance Premium predection API'}

@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaed': model is not None
    }
        
@app.post('/predict')
def predict_premium(data: UserInput):

    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier, 
        'income_lpa': data.income_lpa, 
        'occupation': data.occupation
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={'predicted_category': prediction})






    