# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 07 — Retreinamento Batch e Concept Drift
# MAGIC **Objetivo:** simular o ciclo completo de produção:
# MAGIC novos dados chegam → detectar drift → retreinar → comparar com o modelo atual (champion/challenger) → decidir se promove.
# MAGIC
# MAGIC Este é o fluxo que separa um modelo de notebook de um modelo em produção.

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregar dados históricos (modelo atual foi treinado aqui)

# COMMAND ----------

df_historico = pd.read_csv("/dbfs/FileStore/datasets/churn.csv")
print(f"Dataset histórico: {df_historico.shape[0]} registros")
print(f"Churn rate histórico: {df_historico['churn'].mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simular chegada de novo lote de dados
# MAGIC
# MAGIC Em produção este arquivo chegaria via pipeline de dados
# MAGIC (Kafka, S3, ADLS, etc.) periodicamente.

# COMMAND ----------

np.random.seed(99)
N_NOVO = 600

# Simulamos uma mudança no comportamento dos clientes:
# - suporte aumentou (clientes mais insatisfeitos)
# - novos clientes têm tenure menor
tenure         = np.random.randint(1, 36, N_NOVO)           # tenure menor que antes
monthly_charge = np.round(np.random.uniform(20, 120, N_NOVO), 2)
support_calls  = np.random.poisson(2.8, N_NOVO)             # mais chamadas de suporte
num_products   = np.random.randint(1, 5, N_NOVO)
has_contract   = np.random.choice([0, 1], N_NOVO, p=[0.5, 0.5])
late_payments  = np.random.poisson(1.2, N_NOVO)

churn_score = (
    -0.04 * tenure
    + 0.015 * monthly_charge
    + 0.30  * support_calls
    - 0.25  * num_products
    - 0.70  * has_contract
    + 0.40  * late_payments
    + np.random.normal(0, 0.5, N_NOVO)
)

df_novo_lote = pd.DataFrame({
    "tenure": tenure,
    "monthly_charge": monthly_charge,
    "support_calls": support_calls,
    "num_products": num_products,
    "has_contract": has_contract,
    "late_payments": late_payments,
    "churn": (churn_score > 0.2).astype(int),
})

print(f"Novo lote: {df_novo_lote.shape[0]} registros")
print(f"Churn rate novo lote: {df_novo_lote['churn'].mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Detecção de Concept Drift (PSI — Population Stability Index)
# MAGIC
# MAGIC **Concept drift** ocorre quando a distribuição dos dados muda ao longo do tempo.
# MAGIC Se o modelo foi treinado com dados antigos e o comportamento mudou, as previsões
# MAGIC ficam desatualizadas — mesmo sem nenhum erro de código.
# MAGIC
# MAGIC **PSI (Population Stability Index):**
# MAGIC - PSI < 0.10 → distribuição estável, sem ação necessária
# MAGIC - PSI 0.10–0.25 → mudança moderada, monitorar
# MAGIC - PSI > 0.25 → drift significativo, retreinar

# COMMAND ----------

def calcular_psi(referencia: pd.Series, atual: pd.Series, bins: int = 10) -> float:
    """
    Calcula o PSI entre a distribuição de referência (treino)
    e a distribuição atual (novo lote).
    """
    breakpoints = np.percentile(referencia, np.linspace(0, 100, bins + 1))
    breakpoints  = np.unique(breakpoints)

    ref_counts = np.histogram(referencia, bins=breakpoints)[0]
    cur_counts = np.histogram(atual,      bins=breakpoints)[0]

    ref_pct = ref_counts / len(referencia)
    cur_pct = cur_counts / len(atual)

    # Evitar divisão por zero
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(psi, 4)


FEATURES_MONITORAR = ["tenure", "monthly_charge", "support_calls", "late_payments"]
PSI_THRESHOLD = 0.10

print("=== Análise de Concept Drift (PSI) ===\n")
drift_detectado = False

for feature in FEATURES_MONITORAR:
    psi = calcular_psi(df_historico[feature], df_novo_lote[feature])
    status = "🔴 DRIFT" if psi > 0.25 else "🟡 ATENÇÃO" if psi > PSI_THRESHOLD else "🟢 ESTÁVEL"
    print(f"  {feature:<20} PSI: {psi:.4f}  {status}")
    if psi > PSI_THRESHOLD:
        drift_detectado = True

print(f"\n{'→ Drift detectado — retreinamento recomendado.' if drift_detectado else '→ Distribuições estáveis.'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Treinar modelo champion (atual)
# MAGIC
# MAGIC O champion é o modelo que está em produção hoje.
# MAGIC Treinamos com os dados históricos para ter a baseline de comparação.

# COMMAND ----------

def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charge_per_month"]  = df["monthly_charge"] / (df["tenure"] + 1)
    df["calls_per_product"] = df["support_calls"]  / (df["num_products"] + 1)
    return df

FEATURES = [
    "tenure", "monthly_charge", "support_calls",
    "num_products", "has_contract", "late_payments",
    "charge_per_month", "calls_per_product",
]

def construir_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), FEATURES)], remainder="drop"
        )),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ])

# Champion: treinado com dados históricos
df_hist_fe = preparar_features(df_historico)
X_hist = df_hist_fe[FEATURES]
y_hist = df_hist_fe["churn"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_hist, y_hist, test_size=0.2, random_state=42, stratify=y_hist
)

mlflow.set_experiment("/ml-course/retrain-champion-challenger")

with mlflow.start_run(run_name="champion_v1"):
    pipeline_champion = construir_pipeline()
    pipeline_champion.fit(X_train_c, y_train_c)

    y_prob_c  = pipeline_champion.predict_proba(X_test_c)[:, 1]
    roc_c     = roc_auc_score(y_test_c, y_prob_c)
    pr_c      = average_precision_score(y_test_c, y_prob_c)
    n_treino_c = len(X_train_c)

    mlflow.log_param("model_type",    "random_forest")
    mlflow.log_param("n_estimators",  200)
    mlflow.log_param("n_train",       n_treino_c)
    mlflow.log_metric("roc_auc",      roc_c)
    mlflow.log_metric("pr_auc",       pr_c)
    mlflow.log_metric("psi_max",      max([calcular_psi(df_historico[f], df_novo_lote[f]) for f in FEATURES_MONITORAR]))
    mlflow.sklearn.log_model(pipeline_champion, "model", registered_model_name="churn-champion")

print(f"Champion — ROC-AUC: {roc_c:.4f} | PR-AUC: {pr_c:.4f} | Treino: {n_treino_c} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Treinar modelo challenger
# MAGIC
# MAGIC O challenger é retreinado com dados históricos + novo lote.
# MAGIC Só faz sentido promovê-lo se superar o champion em métricas.

# COMMAND ----------

# Challenger: histórico + novo lote
df_completo = pd.concat([df_historico, df_novo_lote], ignore_index=True)
df_comp_fe  = preparar_features(df_completo)

X_comp = df_comp_fe[FEATURES]
y_comp = df_comp_fe["churn"]

X_train_ch, X_test_ch, y_train_ch, y_test_ch = train_test_split(
    X_comp, y_comp, test_size=0.2, random_state=42, stratify=y_comp
)

with mlflow.start_run(run_name="challenger_retrain_v2"):
    pipeline_challenger = construir_pipeline()
    pipeline_challenger.fit(X_train_ch, y_train_ch)

    y_prob_ch = pipeline_challenger.predict_proba(X_test_ch)[:, 1]
    roc_ch    = roc_auc_score(y_test_ch, y_prob_ch)
    pr_ch     = average_precision_score(y_test_ch, y_prob_ch)
    n_treino_ch = len(X_train_ch)

    mlflow.log_param("model_type",    "random_forest")
    mlflow.log_param("n_estimators",  200)
    mlflow.log_param("n_train",       n_treino_ch)
    mlflow.log_param("inclui_novo_lote", True)
    mlflow.log_metric("roc_auc",      roc_ch)
    mlflow.log_metric("pr_auc",       pr_ch)
    mlflow.sklearn.log_model(pipeline_challenger, "model", registered_model_name="churn-challenger")

print(f"Challenger — ROC-AUC: {roc_ch:.4f} | PR-AUC: {pr_ch:.4f} | Treino: {n_treino_ch} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Decisão Champion vs Challenger
# MAGIC
# MAGIC O challenger só substitui o champion se for **estatisticamente melhor**.
# MAGIC Uma diferença de 0.001 em ROC-AUC não justifica trocar o modelo em produção.

# COMMAND ----------

MARGEM_MINIMA = 0.01  # challenger precisa ser >= 1pp melhor

delta_roc = roc_ch - roc_c
delta_pr  = pr_ch  - pr_c

print("=== Champion vs Challenger ===\n")
print(f"  {'Métrica':<15} {'Champion':>10} {'Challenger':>12} {'Delta':>10}")
print(f"  {'-'*50}")
print(f"  {'ROC-AUC':<15} {roc_c:>10.4f} {roc_ch:>12.4f} {delta_roc:>+10.4f}")
print(f"  {'PR-AUC':<15} {pr_c:>10.4f} {pr_ch:>12.4f} {delta_pr:>+10.4f}")
print(f"  {'N treino':<15} {n_treino_c:>10,} {n_treino_ch:>12,}")
print()

if delta_roc >= MARGEM_MINIMA:
    decisao = "PROMOVER"
    motivo  = f"Challenger supera champion em {delta_roc:+.4f} ROC-AUC (≥ margem mínima de {MARGEM_MINIMA})"
else:
    decisao = "MANTER CHAMPION"
    motivo  = f"Delta de {delta_roc:+.4f} abaixo da margem mínima de {MARGEM_MINIMA} — não justifica troca"

print(f"Decisão: {decisao}")
print(f"Motivo:  {motivo}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Promoção para Production (com revisão humana)
# MAGIC
# MAGIC **Por que não automatizar?**
# MAGIC
# MAGIC Promoção automática parece eficiente mas tem riscos reais:
# MAGIC - O novo lote pode ter dados corrompidos que inflaram as métricas
# MAGIC - A mudança de comportamento pode ser temporária (sazonalidade, evento pontual)
# MAGIC - Em setores regulados (crédito, saúde), promoção automática viola compliance
# MAGIC
# MAGIC O padrão do mercado é: decisão automática de "retreinar", promoção com aprovação humana.

# COMMAND ----------

if decisao == "PROMOVER":
    print("Próximos passos (executar manualmente após revisão):\n")
    print("  1. Revisar os runs no MLflow UI e confirmar as métricas")
    print("  2. Executar testes adicionais no ambiente de staging")
    print("  3. Aprovar a promoção no Model Registry:")
    print()
    print("     client = mlflow.tracking.MlflowClient()")
    print("     client.transition_model_version_stage(")
    print("         name='churn-challenger',")
    print("         version='<version>',")
    print("         stage='Production'")
    print("     )")
    print()
    print("  4. Arquivar a versão anterior do champion:")
    print("     client.transition_model_version_stage(")
    print("         name='churn-champion',")
    print("         version='<version_anterior>',")
    print("         stage='Archived'")
    print("     )")
else:
    print("Champion mantido em produção.")
    print("Novo lote será incluído no próximo ciclo de retreinamento agendado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo do ciclo completo
# MAGIC
# MAGIC ```
# MAGIC Dados históricos
# MAGIC     ↓
# MAGIC Treino champion  →  MLflow Registry (Production)
# MAGIC                            ↓
# MAGIC Novo lote chega   →  Detecção de drift (PSI)
# MAGIC     ↓                      ↓
# MAGIC Retreinamento      Drift detectado?
# MAGIC     ↓                  Sim → retreinar
# MAGIC Challenger             Não → monitorar
# MAGIC     ↓
# MAGIC Champion vs Challenger
# MAGIC     ↓
# MAGIC Revisão humana
# MAGIC     ↓
# MAGIC Promoção para Production
# MAGIC ```
# MAGIC
# MAGIC Este ciclo, em empresas maduras, roda automaticamente via pipelines agendados
# MAGIC (Databricks Workflows, Airflow, Vertex AI Pipelines).
# MAGIC O que muda é a orquestração — a lógica é exatamente esta.
