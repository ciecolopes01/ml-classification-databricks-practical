# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 01 — Modelo base
# MAGIC Construindo o pipeline completo do zero com o dataset de qualidade.

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
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregar e inspecionar os dados

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/datasets/quality.csv")
print(f"Shape: {df.shape}")
print(f"\nTipos:\n{df.dtypes}")
print(f"\nNulos:\n{df.isnull().sum()}")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Definir X e y

# COMMAND ----------

TARGET = "approved"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features: {X.columns.tolist()}")
print(f"\nDistribuição do target:\n{y.value_counts(normalize=True).round(3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Split treino / teste

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} linhas")
print(f"Teste : {X_test.shape[0]} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pipeline profissional

# COMMAND ----------

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

pipeline = Pipeline([
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), numeric_features),
        ],
        remainder="drop",  # descarta colunas não listadas
    )),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Treinar e avaliar

# COMMAND ----------

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Matriz de confusão

# COMMAND ----------

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
plt.title("Matriz de Confusão — Qualidade")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validação cruzada (evita depender de um único split)

# COMMAND ----------

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Feature Importance

# COMMAND ----------

model = pipeline.named_steps["model"]
importances = pd.Series(model.feature_importances_, index=numeric_features)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(6, 4))
importances.plot.barh(ax=ax)
ax.set_title("Importância das features")
ax.set_xlabel("Importância")
plt.tight_layout()
display(fig)
