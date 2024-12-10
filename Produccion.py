import streamlit as st
import joblib
import pandas as pd

# Modelo
Stacking=joblib.load("Stacking.joblib")

#Aplicacion
st.title("Modelo de Producción: Stacking Model")
st.write("Valores requeridos")

#Datos
AñosEstudio=st.number_input("Año de Estudio",min_value=0, step=0.4)
HorasSemanales=st.number_input("Horas Semanales",min_value=0, step=0.4)
NumTrabajadores=st.number_input("Numero de Trabajadores",min_value=0, step=0.4)

#Botón para predecir
if st.button("Predecir"):
  try:
    input=pd.DataFrame([[AñosEstudio,HorasSemanales,NumTrabajadores]],columns=["AñosEstudio: Años de Estudio", "HorasSemanales: Horas Semanales", "NumTrabajadores:N de trabajadores"])
    predict= Stacking.predict(input_data)
    st.success(f"Prediccion: {predict[0]}")
  except Exception as e:
    st.error(f"Error: {e}")
