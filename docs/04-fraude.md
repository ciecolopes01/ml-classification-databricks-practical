# Módulo 04 — Fraude

## Contexto de negócio

Um banco quer detectar transações fraudulentas em tempo real.

O desafio não é técnico — é estatístico: **fraudes são eventos raros**. Em datasets reais, a taxa de fraude costuma ser de 0,1% a 2%. Aqui usamos ~12% para fins didáticos.

---

## O problema do desbalanceamento

Um modelo que prevê "não é fraude" em 100% dos casos teria **~88% de acurácia**.

Isso parece bom. É inútil.

Por isso acurácia não é a métrica certa para fraude.

---

## Métricas corretas para fraude

**Precision (Fraude)** → das transações que bloqueamos, quantas eram realmente fraude?
→ baixa precision = bloquear clientes legítimos (péssima experiência de usuário)

**Recall (Fraude)** → das fraudes reais, quantas detectamos?
→ baixo recall = fraudes passando despercebidas (prejuízo direto)

**PR-AUC (Average Precision)** → resume o trade-off precision/recall para a classe minoritária.
Use esta como métrica principal.

**ROC-AUC** → boa para comparar modelos, mas pode ser enganosa em datasets muito desbalanceados.

---

## Técnicas para tratar desbalanceamento

Neste módulo usamos `class_weight="balanced"`. Outras abordagens existem:

| Técnica | O que faz | Quando usar |
|---|---|---|
| `class_weight="balanced"` | Penaliza erros na classe minoritária | Ponto de partida — simples e eficaz |
| Threshold tuning | Ajusta o ponto de corte (ex: 0.35) | Quando o modelo já é bom mas precision/recall precisam de ajuste |
| SMOTE | Gera amostras sintéticas da minoria | Datasets muito desbalanceados (<1%) |
| Ensemble (BalancedRandomForest) | Combina undersampling + ensemble | Casos extremos |

---

## Ajuste de threshold

O threshold padrão é 0.5 — o modelo prevê fraude se P(fraude) ≥ 0.5.

Baixar para 0.35 aumenta o recall (detecta mais fraudes) mas reduz precision (mais falsos positivos).

A escolha do threshold é uma decisão de negócio, não técnica.

---

## Feature Engineering

**`night_transaction`** → transações entre 0h e 5h têm maior probabilidade de fraude em dados reais.

**`risk_combo`** → merchant desconhecido em país de risco = combinação que amplifica o sinal individual de cada feature.
