import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Union, List
from sklearn.ensemble import RandomForestRegressor

class ReplenishmentModel:

    def __init__(self):
        self._model = None
        self.features_cols = ['gtin_encoded', 'month', 'day', 'dayofweek', 'tipo_E']

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepara los datos crudos para entrenamiento o predicción.
        """
        # Hacer una copia para evitar warnings sobre reescritura de datos
        df = data.copy()

        # 1. Asegurar formato de fecha y extraer componentes temporales
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['month'] = df['fecha'].dt.month
        df['day'] = df['fecha'].dt.day
        df['dayofweek'] = df['fecha'].dt.dayofweek

        # 2. Codificar variables categóricas de forma segura
        # 'gtin' como numérico para el modelo (usamos category codes)
        df['gtin_encoded'] = df['gtin'].astype('category').cat.codes
        
        # 'tipo_movimiento' a One-Hot/Dummy (E = 1, S = 0). Si no existe la columna, lo asumimos como consumo (S = 0)
        if 'tipo_movimiento' in df.columns:
            df['tipo_E'] = (df['tipo_movimiento'] == 'E').astype(int)
        else:
            df['tipo_E'] = 0

        # Guardamos las columnas que usará el modelo
        features = df[['gtin', 'fecha'] + self.features_cols]

        if target_column is not None:
            # Aseguramos que el target sea un DataFrame con la columna solicitada
            target = pd.DataFrame(df[target_column])
            return features, target
        
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Entrena el modelo con los datos preprocesados.
        """
        # Extraemos solo las columnas numéricas que definimos para el entrenamiento
        X = features[self.features_cols]
        y = target.values.ravel() # Aplanamos el target para evitar warnings

        # Inicializamos y entrenamos un regresor potente (RandomForest)
        # Usamos parámetros equilibrados para entrenamiento rápido y robusto
        self._model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        self._model.fit(X, y)

    def _auto_fit_if_needed(self) -> None:
        """
        Entrena el modelo de forma automática si se llama a predict sin un fit previo.
        """
        csv_path = "dataset/movimientos.csv"
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            features, target = self.preprocess(data, target_column="cantidad")
            self.fit(features, target)
        else:
            # Fallback de emergencia si no existiera el dataset en la ruta
            raise ValueError("El modelo no está entrenado y no se encontró 'dataset/movimientos.csv' para auto-entrenamiento.")

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[dict]:
        """
        Predice el consumo para una lista de productos.
        """
        # Si el modelo no ha sido entrenado aún, lo entrenamos automáticamente
        if self._model is None:
            self._auto_fit_if_needed()

        # Tomamos las variables numéricas que corresponden
        X = features[self.features_cols]
        
        # Realizamos la predicción
        preds = self._model.predict(X)
        
        # Cotizamos en 0 si hay consumos negativos
        preds = np.clip(preds, 0, None)

        # Reconstruimos la lista de diccionarios con 'fecha' y 'cantidad'
        fechas = features['fecha'].dt.strftime('%Y-%m-%d').tolist()
        
        results = []
        for f, p in zip(fechas, preds):
            results.append({
                "fecha": f,
                "cantidad": float(p)
            })

        return results

    def save(
        self,
        path: str
    ) -> None:
        """
        Guarda el modelo entrenado en disco.
        """
        if self._model is None:
            self._auto_fit_if_needed()
        
        # Guardamos la estructura del modelo y las columnas esperadas
        model_data = {
            'model': self._model,
            'features_cols': self.features_cols
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(
        self,
        path: str
    ) -> None:
        """
        Carga un modelo entrenado desde disco.
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            self._model = model_data['model']
            self.features_cols = model_data['features_cols']