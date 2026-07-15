import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from datetime import datetime

from challenge.model import ReplenishmentModel

# Inicializar la app y el modelo
app = fastapi.FastAPI()
model = ReplenishmentModel()

# Instanciar y auto-entrenar el modelo al arrancar la API si es necesario
try:
    # Intentamos cargar un modelo guardado o auto-entrenar de inmediato
    model._auto_fit_if_needed()
except Exception as e:
    print(f"Advertencia al inicializar el modelo: {e}")

# 1. Obtener la lista de GTINs válidos del dataset para la validación (HTTP 400)
VALID_GTINS = set()
csv_path = "dataset/movimientos.csv"
if os.path.exists(csv_path):
    try:
        df_mov = pd.read_csv(csv_path)
        VALID_GTINS = set(df_mov['gtin'].astype(str).unique())
    except Exception as e:
        print(f"Error cargando los GTINs válidos: {e}")

# Definición de esquemas de datos usando Pydantic
class ProductInput(BaseModel):
    gtin: str
    fecha: str

class PredictRequest(BaseModel):
    products: List[ProductInput]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(payload: PredictRequest) -> dict:
    # Lista temporal para ir guardando los datos estructurados que le pasaremos al modelo
    parsed_data = []

    for item in payload.products:
        # --- Validación 1: Producto Desconocido (retornar 400) ---
        if item.gtin not in VALID_GTINS:
            raise HTTPException(
                status_code=400, 
                detail=f"El producto con GTIN {item.gtin} es desconocido."
            )

        # --- Validación 2: Fecha Inválida (retornar 400) ---
        try:
            # Validamos que cumpla el formato YYYY-MM-DD
            datetime.strptime(item.fecha, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"La fecha {item.fecha} no tiene un formato válido (YYYY-MM-DD)."
            )

        # Si pasa las validaciones, lo añadimos a nuestra lista
        parsed_data.append({
            "gtin": item.gtin,
            "fecha": item.fecha
        })

    # Convertimos los datos validados a un DataFrame para que el modelo lo entienda
    df_features = pd.DataFrame(parsed_data)

    # Preprocesamos usando el método del modelo
    features = model.preprocess(df_features)

    # Obtenemos las predicciones
    predictions = model.predict(features)

    # Retornamos las predicciones con el formato requerido en la clave "predict"
    return {
        "predict": predictions
    }