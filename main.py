from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API de Churn Prediction", version="1.0")

# Carregar pipeline treinado
model = joblib.load("modelo_pipeline_treinado.pkl")

# Modelo dos dados esperados
class Cliente(BaseModel):
    id_cliente: str
    idade: int
    sexo: str
    cidade: str
    plano_atual: str
    tempo_como_cliente_meses: int
    engajamento_mensal: float
    nps: int
    numero_contatos_suporte: int
    ultima_atividade_dias: int
    qtde_produtos_ativos: int
    renda_mensal: float
    canal_aquisicao: str
    dispositivo_preferido: str
    categoria_profissional: str
    score_interno: float
    uso_cashback: float
    bonus_recebido: int
    interacoes_com_pushs: int
    ticket_medio: float

@app.get("/")
def home():
    return {"mensagem": "API de Previsão de Cancelamento no ar 🔥"}

@app.post("/predict")
def predict(cliente: Cliente):
    try:
        input_df = pd.DataFrame([cliente.dict()])
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        return {
            "probabilidade_cancelamento": round(float(proba), 4),
            "vai_cancelar": bool(pred)
        }
    except Exception as e:
        return {"erro": str(e)}
