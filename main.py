import dill
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from modules.df_reference import isoutlier_3sigma

# Upload reference and client history dfs created earlier

df_reference = pd.read_csv(
    r'data\df_reference.csv',
    parse_dates=['ad_start_tmp'])

df_prev = pd.read_csv(
    r'data\df_previous_activity.csv', dtype={
        'client_id': 'object'}, parse_dates=['prev_visit'])

app = FastAPI()
with open(r'models\sberavto_prediction_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: Union[str, None]
    utm_medium: Union[str, None]
    utm_campaign: Union[str, None]
    utm_adcontent: Union[str, None]
    utm_keyword: Union[str, None]
    device_category: Union[str, None]
    device_os: Union[str, None]
    device_brand: Union[str, None]
    device_model: Union[str, None]
    device_screen_resolution: Union[str, None]
    device_browser: Union[str, None]
    geo_country: Union[str, None]
    geo_city: Union[str, None]


class Prediction(BaseModel):
    session_id: str
    result: int


@app.get('/status')
def status():
    return "I'm ok"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'session_id': form.session_id,
        'result': y[0]
    }
