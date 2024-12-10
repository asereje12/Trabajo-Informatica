from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class IngresoData(BaseModel):  # Usamos 'IngresoData' en lugar de 'Datos'
    AñosEstudio: float
    HorasSemanales: float
    NumTrabajadores: int

app = FastAPI()

# Cargar el modelo de manera simple al iniciar la aplicación
model = pickle.load(open("Stacking.json", "rb"))

@app.get("/")
def index():
    return {
        "msg" : "¡Bienvenidos a la plataforma de Machine Learning!",
        "org": "MLAAS",
        "api-documentation": "https://trabajo-informatica-lstgptqe9drmgwbjj2lr8g.streamlit.app/",
    }

@app.post("/predict")
def get_home_price(data: IngresoData):
    received = data.dict()
    ingreso_attr=[[ 
        received['AñosEstudio'],
        received['HorasSemanales'],
        received['NumTrabajadores'],
    ]]
    ingreso = model.predict(ingreso_attr).tolist()[0]
    return {'data': received, 'ingreso': ingreso}
    st.write("El valor es: ")
