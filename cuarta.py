from typing import Optional
from fastapi import FastAPI
from data_class import y_pred_stacking
import pickle

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = pickle.load(open("Stacking_model.pkl", "rb"))

@app.post("/predict")
def get_home_price(data: y_pred_stacking):
    received = data.dict()
    ingreso_attr=[[
        received['AñosEstudio'],
        received['HorasSemanales'],
        received['NumTrabajadores'],
    ]]
    ingreso=model.predict(ingreso_attr).tolist()[0]
    return{'data':received, 'ingreso':ingreso}
