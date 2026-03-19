# Módulo 05 — Logística

## Contexto de negócio

Uma transportadora quer prever se um pedido será entregue com atraso antes de despachar.

Com a previsão, pode priorizar pedidos de risco, acionar transportadoras alternativas ou avisar o cliente proativamente.

---

## O novo desafio: feature categórica

Os módulos anteriores tinham apenas features numéricas. Aqui introduzimos `carrier` (transportadora A, B ou C).

Modelos de ML não aceitam texto diretamente. Precisamos transformar categorias em números — sem introduzir ordem artificial.

---

## Como tratar categorias no pipeline

Usamos `OrdinalEncoder` dentro do `ColumnTransformer`:

```python
ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(...), CATEGORICAL_FEATURES),
    ]
)
```

O pipeline aplica transformações diferentes por tipo de coluna. Tudo dentro do mesmo `fit/transform` — sem vazamento de dados.

---

## Features do dataset

| Feature | Tipo | Significado |
|---|---|---|
| `distance_km` | numérica | Distância de entrega |
| `weight_kg` | numérica | Peso do pedido |
| `carrier` | categórica | Transportadora (A/B/C) |
| `weather_issue` | binária | Problema climático? |
| `peak_season` | binária | Período de pico? |
| `warehouse_load` | numérica | Ocupação do armazém (0-1) |

---

## Feature Engineering

**`heavy_long`** → peso alto E distância longa combinados aumentam mais o risco do que cada um separado.

**`overloaded_warehouse`** → carga acima de 80% no armazém implica filas, erros operacionais e atrasos sistêmicos.
