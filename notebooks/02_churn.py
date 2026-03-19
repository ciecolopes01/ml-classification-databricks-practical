# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 02 — Churn
# MAGIC **Contexto de negócio:** prever quais clientes vão cancelar o serviço
# MAGIC nos próximos 30 dias para agir preventivamente.

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
    roc_curve,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregar e inspecionar

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/datasets/churn.csv")
print(f"Shape: {df.shape}")
print(f"\nDistribuição do target:\n{df['churn'].value_counts(normalize=True).round(3)}")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Engineering
# MAGIC
# MAGIC Esta é a etapa mais importante.
# MAGIC
# MAGIC Não criamos features aleatórias — cada uma deve ter uma hipótese de negócio.
# MAGIC - `charge_per_month` → cliente paga caro por pouco tempo? Sinal de insatisfação.
# MAGIC - `calls_per_product` → muita chamada para poucos produtos → provável problema.

# COMMAND ----------

df["charge_per_month"]  = df["monthly_charge"] / (df["tenure"] + 1)
df["calls_per_product"] = df["support_calls"]  / (df["num_products"] + 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Definir X e y

# COMMAND ----------

TARGET = "churn"
FEATURES = [
    "tenure", "monthly_charge", "support_calls",
    "num_products", "has_contract", "late_payments",
    "charge_per_month", "calls_per_product",
]

X = df[FEATURES]
y = df[TARGET]

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pipeline

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
        remainder="drop",
    )),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Avaliação

# COMMAND ----------

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Matriz de confusão
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
axes[0].set_title("Matriz de Confusão")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1].set_xlabel("FPR"), axes[1].set_ylabel("TPR")
axes[1].set_title("Curva ROC"), axes[1].legend()

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Importance

# COMMAND ----------

model = pipeline.named_steps["model"]
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
importances.plot.barh(ax=ax)
ax.set_title("Importância das features — Churn")
ax.set_xlabel("Importância")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validação cruzada

# COMMAND ----------

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reflexão
# MAGIC
# MAGIC - O modelo acerta bem? Onde erra mais?
# MAGIC - As features criadas (`charge_per_month`, `calls_per_product`) aparecem com alta importância?
# MAGIC - O que isso diz sobre o comportamento do cliente antes do churn?
