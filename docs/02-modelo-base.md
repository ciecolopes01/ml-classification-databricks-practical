# Módulo 02 — Modelo base

## Objetivo

Construir o pipeline completo do zero, do CSV à avaliação, usando o dataset de qualidade.

Este é o template que todos os outros notebooks reutilizam. Entenda cada etapa aqui e os próximos módulos ficam mais fáceis.

---

## Estrutura do notebook

```
1. Carregar e inspecionar os dados
2. Definir X e y
3. Split treino / teste
4. Construir o Pipeline
5. Treinar e avaliar
6. Matriz de confusão
7. Validação cruzada
8. Feature Importance
```

---

## Por que Random Forest?

Random Forest é o ponto de partida ideal para classificação:

- Funciona bem sem muito ajuste de hiperparâmetros
- Lida com features em escalas diferentes
- Fornece Feature Importance nativamente
- Robusto a outliers

Não é o algoritmo mais poderoso, mas é o mais confiável para começar.

---

## O Pipeline em detalhe

```python
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
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])
```

**`SimpleImputer(strategy="median")`**  
Substitui valores nulos pela mediana da coluna. Mediana é preferível à média por ser menos afetada por outliers.

**`StandardScaler()`**  
Normaliza as features para média 0 e desvio padrão 1. Necessário para algoritmos sensíveis à escala. Random Forest não precisa, mas é boa prática manter no pipeline para quando trocar o modelo.

**`ColumnTransformer`**  
Aplica transformações diferentes por tipo de coluna. O argumento `remainder="drop"` garante que colunas não listadas sejam descartadas — evita surpresas com colunas inesperadas.

**`remainder="drop"` — por que é importante**  
Sem ele, colunas não tratadas passam direto para o modelo. Se o dataset tiver uma coluna de ID ou texto, o modelo vai tentar usá-la como feature numérica e quebrar silenciosamente.

---

## Validação cruzada

```python
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

Com um único `train_test_split`, o resultado pode ter sorte ou azar dependendo de quais exemplos caíram no teste. Com 5 folds, avaliamos o modelo em 5 subconjuntos diferentes e reportamos a média e o desvio padrão.

Um bom modelo tem desvio padrão baixo (< 0.02) — significa que performa consistentemente independente do split.

---

## Feature Importance

Feature Importance do Random Forest mede quanto cada feature contribuiu para as decisões das árvores.

```python
importances = pd.Series(
    pipeline.named_steps["model"].feature_importances_,
    index=numeric_features
).sort_values(ascending=True)
```

**Como interpretar:**
- Features com importância alta → o modelo depende muito delas
- Features com importância próxima de zero → provavelmente podem ser removidas
- Se uma feature que faz sentido de negócio tem importância zero, investigue — pode estar com problema na engenharia

Feature Importance não explica o **sinal** da relação (positivo ou negativo), apenas o **peso**. Para entender direção, use SHAP (avançado, fora do escopo deste curso).
