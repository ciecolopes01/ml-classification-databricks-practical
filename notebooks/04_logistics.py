# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 04 — Logística
# MAGIC **Contexto de negócio:** prever se um pedido será entregue com atraso.
# MAGIC
# MAGIC **Desafio principal:** features categóricas (transportadora).
# MAGIC Aqui introduzimos o `OrdinalEncoder` dentro do pipeline.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/datasets/logistics.csv")
print(f"Shape: {df.shape}")
print(f"\nTransportadoras:\n{df['carrier'].value_counts()}")
print(f"\nTaxa de atraso: {df['delayed'].mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Pedidos pesados E distantes têm mais chance de atraso
df["heavy_long"] = (df["weight_kg"] > 10).astype(int) * (df["distance_km"] > 500).astype(int)

# Carga no armazém acima de 80% = sobrecarga
df["overloaded_warehouse"] = (df["warehouse_load"] > 0.80).astype(int)

# COMMAND ----------

TARGET = "delayed"
NUMERIC_FEATURES = [
    "distance_km", "weight_kg", "weather_issue",
    "peak_season", "warehouse_load",
    "heavy_long", "overloaded_warehouse",
]
CATEGORICAL_FEATURES = ["carrier"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline com features numéricas e categóricas

# COMMAND ----------

pipeline = Pipeline([
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

# COMMAND ----------

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["no prazo", "atrasado"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# COMMAND ----------

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["No prazo", "Atrasado"], ax=ax
)
plt.title("Matriz de Confusão — Logística")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

model = pipeline.named_steps["model"]
all_feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
importances = pd.Series(model.feature_importances_, index=all_feature_names)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
importances.plot.barh(ax=ax)
ax.set_title("Importância das features — Logística")
plt.tight_layout()
display(fig)

# COMMAND ----------

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
