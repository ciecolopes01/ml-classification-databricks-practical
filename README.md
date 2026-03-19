# Machine Learning na prática (sem enrolação)
### Classificação real com Python + Databricks + MLOps

> Você não aprende Machine Learning fazendo `.fit()`.  
> Você aprende entendendo dados, decisões e como operar modelos.

---

## O que você vai aprender

Este mini-curso é 100% prático. Você vai executar passo a passo:

- Machine Learning de verdade (sem teoria inútil)
- Classificação com contexto de negócio
- Feature Engineering com hipóteses reais
- Pipeline profissional com scikit-learn
- Tratar desbalanceamento de classes
- Evitar data leakage
- MLOps com MLflow (tracking, registro, versionamento)

---

## Casos reais (hands-on)

Você vai construir modelos para 4 problemas de negócio:

| # | Problema | Desafio técnico |
|---|---|---|
| 1 | Churn — cliente vai cancelar? | Feature Engineering, Curva ROC |
| 2 | Fraude — transação suspeita? | Desbalanceamento, PR-AUC, threshold |
| 3 | Logística — pedido vai atrasar? | Features categóricas no pipeline |
| 4 | Qualidade — produto aprovado? | Pipeline base, Feature Importance |

---

## Stack

- Python 3.10+
- Databricks Free Edition
- scikit-learn
- MLflow
- matplotlib

---

## Estrutura do curso

```
ml-classification-databricks/
│
├── README.md
│
├── docs/                          # Leia antes de executar cada notebook
│   ├── 00-setup.md
│   ├── 01-fundamentos.md
│   ├── 02-modelo-base.md
│   ├── 03-churn.md
│   ├── 04-fraude.md
│   ├── 05-logistica.md
│   ├── 06-qualidade.md
│   ├── 07-mlops.md
│   └── 08-mentalidade.md
│
├── notebooks/                     # Execute em ordem no Databricks
│   ├── 01_base_model.py           # Pipeline completo + validação cruzada
│   ├── 02_churn.py                # Feature Engineering + Curva ROC
│   ├── 03_fraud.py                # Desbalanceamento + PR-AUC + threshold
│   ├── 04_logistics.py            # Features categóricas no pipeline
│   └── 05_mlops.py                # MLflow: tracking + registro de modelos
│
├── datasets/
│   ├── churn.csv
│   ├── fraud.csv
│   ├── logistics.csv
│   └── quality.csv
│
└── requirements.txt
```

---

## Como usar

1. Leia `docs/00-setup.md` e configure seu ambiente Databricks
2. Faça upload dos datasets na pasta `/dbfs/FileStore/datasets/`
3. Execute os notebooks em ordem
4. Leia o doc correspondente **antes** de cada notebook
5. Não copie e cole — entenda cada etapa

---

## Diferencial

Este curso não é sobre ferramenta. É sobre:

- Pensar ML com contexto de negócio
- Construir features com hipóteses reais
- Escolher métricas certas para cada problema
- Evitar os erros que modelos em produção cometem

> Você não vai sair sabendo "rodar código".  
> Vai sair entendendo como ML funciona de verdade.

---

## Acesso

Este repositório é privado.  
Acesso liberado após aquisição em: **[link-de-venda]**
