# Módulo 06 — MLOps com MLflow

## O que é MLOps?

MLOps é o conjunto de práticas para levar modelos de ML do notebook para produção — e mantê-los funcionando ao longo do tempo.

Sem MLOps, modelos existem apenas no seu computador.

---

## Por que MLflow?

Sem tracking de experimentos, você não sabe:
- Qual versão do modelo está em produção
- Quais parâmetros geraram aquele resultado
- Por que o modelo piorou depois de um retrain

MLflow resolve os três problemas.

---

## Conceitos principais

**Experiment** → agrupa runs relacionados. Ex: todos os treinos do modelo de churn.

**Run** → uma execução de treinamento. Cada run registra: parâmetros, métricas e artefatos.

**Model Registry** → cataloga versões de modelos com status (Staging / Production / Archived).

---

## O que registrar em cada run

| O que | Por quê |
|---|---|
| Parâmetros do modelo | Reproduzir o experimento |
| Métricas (ROC-AUC, PR-AUC) | Comparar runs |
| Artefato do modelo | Carregar em produção |
| Tamanho do dataset | Detectar mudanças nos dados |

---

## Fluxo completo

```
Notebook (treino)
    → mlflow.start_run()
        → log_params()
        → log_metrics()
        → log_model()
    → Model Registry
        → Staging → testes
        → Production → serve via API
```

---

## Próximos passos além deste curso

- **Model monitoring:** detectar quando o modelo degrada em produção
- **Feature store:** centralizar features reutilizáveis entre modelos
- **CI/CD para ML:** automatizar retreinamento quando dados mudam
