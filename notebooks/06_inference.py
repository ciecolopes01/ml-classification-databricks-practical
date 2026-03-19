# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 06 — Inferência Online
# MAGIC **Objetivo:** usar o modelo treinado para prever em tempo real,
# MAGIC simulando o que acontece quando um novo registro chega via API.
# MAGIC
# MAGIC Em produção, este fluxo é chamado milhares de vezes por segundo.
# MAGIC Aqui simulamos um registro por vez para entender cada etapa.

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregar o modelo do MLflow Registry
# MAGIC
# MAGIC Em produção, o modelo é carregado uma vez na inicialização do serviço
# MAGIC e reutilizado para todas as requisições subsequentes.

# COMMAND ----------

# Opção A: carregar pelo nome registrado (produção)
MODEL_NAME    = "churn-random_forest"
MODEL_VERSION = "latest"

try:
    model_uri      = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    loaded_pipeline = mlflow.sklearn.load_model(model_uri)
    print(f"Modelo carregado do Registry: {model_uri}")

except Exception:
    # Opção B: treinar localmente se o Registry não estiver disponível
    print("Registry não disponível — treinando modelo local para demonstração...")

    df = pd.read_csv("/dbfs/FileStore/datasets/churn.csv")
    df["charge_per_month"]  = df["monthly_charge"] / (df["tenure"] + 1)
    df["calls_per_product"] = df["support_calls"]  / (df["num_products"] + 1)

    FEATURES = [
        "tenure", "monthly_charge", "support_calls",
        "num_products", "has_contract", "late_payments",
        "charge_per_month", "calls_per_product",
    ]
    X = df[FEATURES]
    y = df["churn"]

    loaded_pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), FEATURES)], remainder="drop"
        )),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    loaded_pipeline.fit(X, y)
    print("Modelo local treinado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inferência online — um registro por vez
# MAGIC
# MAGIC Simula a chegada de um novo cliente via requisição de API.
# MAGIC O modelo retorna a probabilidade de churn em milissegundos.

# COMMAND ----------

def predict_churn(cliente: dict, threshold: float = 0.40) -> dict:
    """
    Recebe um dicionário com os dados do cliente,
    calcula features derivadas e retorna a previsão.
    """
    df_input = pd.DataFrame([cliente])

    # Mesmas features derivadas do treino — crítico manter consistência
    df_input["charge_per_month"]  = df_input["monthly_charge"] / (df_input["tenure"] + 1)
    df_input["calls_per_product"] = df_input["support_calls"]  / (df_input["num_products"] + 1)

    FEATURES = [
        "tenure", "monthly_charge", "support_calls",
        "num_products", "has_contract", "late_payments",
        "charge_per_month", "calls_per_product",
    ]

    start = time.time()
    prob  = loaded_pipeline.predict_proba(df_input[FEATURES])[0][1]
    latencia_ms = (time.time() - start) * 1000

    return {
        "prob_churn":   round(prob, 4),
        "churn":        int(prob >= threshold),
        "risco":        "ALTO" if prob >= 0.6 else "MÉDIO" if prob >= threshold else "BAIXO",
        "threshold":    threshold,
        "latencia_ms":  round(latencia_ms, 2),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Caso 1 — Cliente de alto risco

# COMMAND ----------

cliente_risco = {
    "tenure": 2,
    "monthly_charge": 110.0,
    "support_calls": 7,
    "num_products": 1,
    "has_contract": 0,
    "late_payments": 3,
}

resultado = predict_churn(cliente_risco)
print("Cliente de ALTO RISCO:")
for k, v in resultado.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Caso 2 — Cliente fidelizado

# COMMAND ----------

cliente_fiel = {
    "tenure": 48,
    "monthly_charge": 45.0,
    "support_calls": 0,
    "num_products": 4,
    "has_contract": 1,
    "late_payments": 0,
}

resultado = predict_churn(cliente_fiel)
print("Cliente FIDELIZADO:")
for k, v in resultado.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inferência em lote — múltiplos registros
# MAGIC
# MAGIC Quando um arquivo de clientes chega diariamente para scoring,
# MAGIC processamos todos de uma vez (muito mais eficiente que um por um).

# COMMAND ----------

# Simula arquivo diário de clientes para scoring
clientes_para_score = pd.DataFrame([
    {"tenure": 2,  "monthly_charge": 110.0, "support_calls": 7, "num_products": 1, "has_contract": 0, "late_payments": 3},
    {"tenure": 48, "monthly_charge":  45.0, "support_calls": 0, "num_products": 4, "has_contract": 1, "late_payments": 0},
    {"tenure": 12, "monthly_charge":  80.0, "support_calls": 3, "num_products": 2, "has_contract": 0, "late_payments": 1},
    {"tenure": 36, "monthly_charge":  60.0, "support_calls": 1, "num_products": 3, "has_contract": 1, "late_payments": 0},
    {"tenure": 1,  "monthly_charge":  95.0, "support_calls": 5, "num_products": 1, "has_contract": 0, "late_payments": 2},
])

# Feature engineering em lote
clientes_para_score["charge_per_month"]  = clientes_para_score["monthly_charge"] / (clientes_para_score["tenure"] + 1)
clientes_para_score["calls_per_product"] = clientes_para_score["support_calls"]  / (clientes_para_score["num_products"] + 1)

FEATURES = [
    "tenure", "monthly_charge", "support_calls",
    "num_products", "has_contract", "late_payments",
    "charge_per_month", "calls_per_product",
]

THRESHOLD = 0.40

clientes_para_score["prob_churn"] = loaded_pipeline.predict_proba(clientes_para_score[FEATURES])[:, 1]
clientes_para_score["churn_pred"] = (clientes_para_score["prob_churn"] >= THRESHOLD).astype(int)
clientes_para_score["risco"]      = clientes_para_score["prob_churn"].apply(
    lambda p: "ALTO" if p >= 0.6 else "MÉDIO" if p >= THRESHOLD else "BAIXO"
)

display(clientes_para_score[["tenure", "monthly_charge", "support_calls",
                              "prob_churn", "churn_pred", "risco"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validação de entrada (input validation)
# MAGIC
# MAGIC Em produção, dados chegam sujos. Validar antes de passar ao modelo
# MAGIC evita erros silenciosos e previsões sem sentido.

# COMMAND ----------

def validar_cliente(cliente: dict) -> tuple[bool, list]:
    erros = []

    campos_obrigatorios = [
        "tenure", "monthly_charge", "support_calls",
        "num_products", "has_contract", "late_payments"
    ]
    for campo in campos_obrigatorios:
        if campo not in cliente or cliente[campo] is None:
            erros.append(f"Campo ausente ou nulo: {campo}")

    if "tenure" in cliente and cliente["tenure"] < 0:
        erros.append("tenure não pode ser negativo")
    if "monthly_charge" in cliente and cliente["monthly_charge"] <= 0:
        erros.append("monthly_charge deve ser maior que zero")

    return len(erros) == 0, erros


# Teste com registro inválido
cliente_invalido = {
    "tenure": -5,
    "monthly_charge": None,
    "support_calls": 2,
    "num_products": 1,
    "has_contract": 0,
    "late_payments": 1,
}

valido, erros = validar_cliente(cliente_invalido)
if not valido:
    print("Registro rejeitado:")
    for erro in erros:
        print(f"  → {erro}")
else:
    resultado = predict_churn(cliente_invalido)
    print(resultado)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pontos-chave
# MAGIC
# MAGIC - O modelo é carregado **uma vez** e reutilizado para todas as previsões
# MAGIC - Feature engineering deve ser **idêntico** ao aplicado no treino — qualquer diferença gera previsões erradas
# MAGIC - Threshold é uma decisão de negócio, não do modelo — ajuste conforme o custo de cada tipo de erro
# MAGIC - Validação de entrada evita que dados corrompidos cheguem ao modelo silenciosamente
# MAGIC - Latência de inferência com Random Forest é tipicamente < 5ms por registro
