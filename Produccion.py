import streamlit as st
import joblib
import pandas as pd

# Modelo
Stacking=joblib.load("Stacking.joblib")