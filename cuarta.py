from typing import Optional
from fastapi import FastAPI
from data_class import Datos
import pickle

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = pickle.load(open("Stacking_model.pkl", "rb"))

@app.get("/")
def index():
    return {
        "msg" : "¡Bienvenidos a la plataforma de Machine Learning!",
        "org": "MLAAS",
        "api-documentation": "https://trabajo-informatica-hmwexyjfxucjubtrxjp5v4.streamlit.app/",
    }

@app.post("/predict")
def get_home_price(data: Datos):
    received = data.dict()
    ingreso_attr=[[
        received['AñosEstudio'],
        received['HorasSemanales'],
        received['NumTrabajadores'],
    ]]
    ingreso=model.predict(ingreso_attr).tolist()[0]
    return{'data':received, 'ingreso':ingreso}
