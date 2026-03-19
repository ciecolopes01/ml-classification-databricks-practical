# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 05 — MLOps com MLflow
# MAGIC **Objetivo:** rastrear experimentos, comparar modelos e registrar o melhor.
# MAGIC
# MAGIC MLOps começa aqui: sem tracking, você não sabe qual modelo está em produção
# MAGIC nem como ele foi treinado.

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dados (usamos churn — tem features variadas)

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/datasets/churn.csv")

df["charge_per_month"]  = df["monthly_charge"] / (df["tenure"] + 1)
df["calls_per_product"] = df["support_calls"]  / (df["num_products"] + 1)

TARGET = "churn"
FEATURES = [
    "tenure", "monthly_charge", "support_calls",
    "num_products", "has_contract", "late_payments",
    "charge_per_month", "calls_per_product",
]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Função auxiliar: construir pipeline

# COMMAND ----------

def build_pipeline(model):
    numeric_features = X.columns.tolist()
    return Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",  StandardScaler()),
                ]), numeric_features),
            ],
            remainder="drop",
        )),
        ("model", model),
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Experimento MLflow — comparar 3 modelos

# COMMAND ----------

mlflow.set_experiment("/ml-course/churn-comparison")

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
}

results = {}

for model_name, model in MODELS.items():
    with mlflow.start_run(run_name=model_name):

        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)

        # Logar parâmetros do modelo
        mlflow.log_params(model.get_params())

        # Logar métricas
        mlflow.log_metric("roc_auc",  roc_auc)
        mlflow.log_metric("pr_auc",   pr_auc)
        mlflow.log_metric("n_train",  X_train.shape[0])
        mlflow.log_metric("n_test",   X_test.shape[0])

        # Logar o modelo
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=f"churn-{model_name}",
        )

        results[model_name] = {"roc_auc": roc_auc, "pr_auc": pr_auc}
        print(f"{model_name:<25} ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comparar resultados

# COMMAND ----------

results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
print("\n=== Ranking dos modelos ===")
display(results_df)

best_model = results_df.index[0]
print(f"\nMelhor modelo: {best_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Carregar o modelo registrado (simula produção)

# COMMAND ----------

model_uri = f"models:/churn-{best_model}/latest"

try:
    loaded_pipeline = mlflow.sklearn.load_model(model_uri)
    y_pred_loaded   = loaded_pipeline.predict(X_test)
    roc_loaded      = roc_auc_score(y_test, loaded_pipeline.predict_proba(X_test)[:, 1])
    print(f"Modelo carregado do Registry: {model_uri}")
    print(f"ROC-AUC (modelo carregado): {roc_loaded:.4f}")
except Exception as e:
    print(f"Registry não disponível neste ambiente: {e}")
    print("Em produção: use o URI acima para carregar qualquer versão registrada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Por que isso importa
# MAGIC
# MAGIC Sem MLflow:
# MAGIC - Você não sabe qual modelo está em produção
# MAGIC - Não consegue comparar experimentos
# MAGIC - Não tem como reproduzir um resultado
# MAGIC
# MAGIC Com MLflow:
# MAGIC - Todo experimento é rastreado automaticamente
# MAGIC - Você compara runs em uma UI visual
# MAGIC - O modelo em produção tem versão, métricas e parâmetros registrados
