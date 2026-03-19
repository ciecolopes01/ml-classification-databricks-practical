# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 03 — Fraude
# MAGIC **Contexto de negócio:** detectar transações fraudulentas.
# MAGIC
# MAGIC **Desafio principal:** o dataset é desbalanceado — fraudes são raras.
# MAGIC Um modelo ingênuo que prevê "não fraude" sempre teria ~88% de acurácia
# MAGIC e seria completamente inútil. Aqui aprendemos a lidar com isso.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregar e entender o desbalanceamento

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/datasets/fraud.csv")

print("Distribuição do target:")
print(df["fraud"].value_counts())
print(f"\n→ Taxa de fraude: {df['fraud'].mean():.1%}")
print("\nUm modelo que sempre diz 'não fraude' teria acurácia de "
      f"{1 - df['fraud'].mean():.1%} — e seria inútil.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Engineering
# MAGIC
# MAGIC - `night_transaction` → transações de madrugada são mais suspeitas
# MAGIC - `risk_combo` → novo merchant + país de risco = sinal forte

# COMMAND ----------

df["night_transaction"] = (df["hour"].between(0, 5)).astype(int)
df["risk_combo"]        = df["is_new_merchant"] * (df["country_risk"] > 0).astype(int)

# COMMAND ----------

TARGET = "fraud"
FEATURES = [
    "amount", "hour", "distance_home_km",
    "is_new_merchant", "repeat_tries", "country_risk",
    "night_transaction", "risk_combo",
]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Modelo sem tratamento de desbalanceamento (baseline ingênuo)

# COMMAND ----------

numeric_features = X.columns.tolist()

def build_pipeline(class_weight=None, threshold=0.5):
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
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight,
        )),
    ])

# Pipeline sem tratamento
pipeline_naive = build_pipeline(class_weight=None)
pipeline_naive.fit(X_train, y_train)
y_pred_naive = pipeline_naive.predict(X_test)
y_prob_naive = pipeline_naive.predict_proba(X_test)[:, 1]

print("=== Modelo INGÊNUO (sem tratamento) ===")
print(classification_report(y_test, y_pred_naive, target_names=["legítima", "fraude"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Modelo com class_weight="balanced"

# COMMAND ----------

pipeline_balanced = build_pipeline(class_weight="balanced")
pipeline_balanced.fit(X_train, y_train)
y_pred_balanced = pipeline_balanced.predict(X_test)
y_prob_balanced = pipeline_balanced.predict_proba(X_test)[:, 1]

print("=== Modelo BALANCEADO (class_weight='balanced') ===")
print(classification_report(y_test, y_pred_balanced, target_names=["legítima", "fraude"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Comparação de métricas corretas para fraude
# MAGIC
# MAGIC Para datasets desbalanceados, a métrica principal **não é acurácia**.
# MAGIC Usamos:
# MAGIC - **ROC-AUC** — separabilidade geral
# MAGIC - **PR-AUC (Average Precision)** — foco na classe minoritária (fraude)

# COMMAND ----------

print("Comparação de modelos:")
print(f"{'Métrica':<25} {'Ingênuo':>12} {'Balanceado':>12}")
print("-" * 50)
print(f"{'ROC-AUC':<25} {roc_auc_score(y_test, y_prob_naive):>12.4f} "
      f"{roc_auc_score(y_test, y_prob_balanced):>12.4f}")
print(f"{'PR-AUC (fraude)':<25} {average_precision_score(y_test, y_prob_naive):>12.4f} "
      f"{average_precision_score(y_test, y_prob_balanced):>12.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Precision-Recall Curve (a mais importante para fraude)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, y_prob, label in [
    (axes[0], y_prob_naive,    "Ingênuo"),
    (axes[1], y_prob_balanced, "Balanceado"),
]:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall"), ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {label}")
    ax.legend()

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Ajuste de threshold
# MAGIC
# MAGIC O threshold padrão é 0.5. Para fraude, pode ser melhor abaixar
# MAGIC (capturar mais fraudes mesmo com mais falsos positivos).

# COMMAND ----------

threshold = 0.35
y_pred_thresh = (y_prob_balanced >= threshold).astype(int)

print(f"=== Threshold = {threshold} ===")
print(classification_report(y_test, y_pred_thresh, target_names=["legítima", "fraude"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reflexão
# MAGIC
# MAGIC - Por que acurácia é enganosa em datasets desbalanceados?
# MAGIC - O `class_weight="balanced"` melhorou recall de fraude?
# MAGIC - Qual threshold faz mais sentido para o negócio?
# MAGIC   (Custo de uma fraude não detectada vs. custo de bloquear cliente legítimo)
