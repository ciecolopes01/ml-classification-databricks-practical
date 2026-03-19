# Módulo 01 — Fundamentos

## O que é Machine Learning?

É o processo de encontrar padrões em dados históricos e usar esses padrões para fazer previsões em dados novos.

O modelo não é programado com regras. Ele aprende as regras a partir dos exemplos.

---

## O que é classificação?

Classificação é prever a qual **categoria** um exemplo pertence.

| Problema | Categorias |
|---|---|
| Churn | cancelou / não cancelou |
| Fraude | fraude / legítima |
| Logística | atrasou / no prazo |
| Qualidade | aprovado / reprovado |

Todos os problemas deste curso são classificação binária: duas categorias possíveis.

---

## Componentes principais

**Features (X)**  
São as variáveis de entrada — o que o modelo usa para aprender e prever.  
Exemplo: `tenure`, `monthly_charge`, `support_calls`.

**Target (y)**  
É o que queremos prever — a variável de saída.  
Exemplo: `churn` (0 ou 1).

**Modelo**  
É a função matemática que transforma X em y.  
Neste curso usamos Random Forest como modelo principal.

---

## Feature Engineering

É a etapa de criar ou transformar features para torná-las mais úteis ao modelo.

É o ponto mais importante de qualquer projeto de ML. Um modelo simples com boas features supera um modelo complexo com features ruins.

**Regra:** cada feature criada deve ter uma hipótese de negócio.

```python
# hipótese: cliente que paga caro por pouco tempo tem mais chance de cancelar
df["charge_per_month"] = df["monthly_charge"] / (df["tenure"] + 1)
```

---

## Data Leakage

É o erro mais crítico em ML. Acontece quando o modelo usa informações que não estariam disponíveis no momento da previsão real.

**Exemplo de leakage em churn:**  
Usar `dias_sem_acesso` medido depois que o cliente já cancelou. No momento real da previsão, essa informação não existe.

**Consequência:** o modelo parece excelente no teste, mas falha completamente em produção.

**Como evitar:** pergunte sempre — "essa informação existiria no momento em que precisamos fazer a previsão?"

---

## Split treino / teste

Avaliamos o modelo em dados que ele nunca viu durante o treino.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- `test_size=0.2` → 20% para teste, 80% para treino
- `stratify=y` → mantém a proporção de classes igual nos dois conjuntos
- `random_state=42` → resultado reproduzível

---

## Pipeline

Pipeline encadeia pré-processamento e modelo em um único objeto.

```
dados brutos → SimpleImputer → StandardScaler → RandomForest → previsão
```

**Por que usar Pipeline:**
- Evita data leakage: o scaler aprende apenas com os dados de treino
- Código mais limpo e reproduzível
- Em produção, um único objeto recebe dados brutos e retorna previsão

---

## Métricas de avaliação

**Accuracy** → percentual de acertos. Enganosa em datasets desbalanceados.

**Precision** → dos que previmos como positivo, quantos realmente são?

**Recall** → dos que realmente são positivos, quantos identificamos?

**ROC-AUC** → capacidade geral de separar as classes. Entre 0 e 1, quanto maior melhor. 0.5 = aleatório.

**PR-AUC** → foco na classe minoritária. Usar quando o dataset é desbalanceado.
