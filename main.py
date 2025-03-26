from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API de Churn Prediction", version="1.0")

model = joblib.load("modelo_pipeline_treinado.pkl")  # Suba esse arquivo depois

class Cliente(BaseModel):
    idade: int
    tempo_como_cliente_meses: int
    renda_mensal: float
    engajamento_mensal: float
    ultima_atividade_dias: int
    interacoes_com_pushs: int
    numero_contatos_suporte: int
    ticket_medio: float
    nps: float
    uso_cashback: int
    bonus_recebido: int
    score_interno: float
    plano_atual: str
    canal_aquisicao: str
    categoria_profissional: str
    cidade: str
    dispositivo_preferido: str
    sexo: str

@app.get("/")
def home():
    return {"mensagem": "API de Previsão de Cancelamento no ar 🔥"}

@app.post("/predict")
def predict(cliente: Cliente):
    input_data = pd.DataFrame([cliente.dict()])
    proba = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]
    return {
        "probabilidade_cancelamento": round(float(proba), 4),
        "vai_cancelar": bool(pred)
    }
