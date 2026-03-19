# Módulo 06 — Qualidade de produto

## Contexto de negócio

Uma vinícola quer automatizar a aprovação de lotes com base em análises físico-químicas, reduzindo a dependência de sommelier para triagem inicial.

---

## Por que este dataset é diferente

O dataset de qualidade é o mais "real" do curso: as features têm correlações físicas entre si (`density` correlaciona com `alcohol` e `residual_sugar`). Isso torna a Feature Importance mais interessante de interpretar.

---

## Features

| Feature | Descrição |
|---|---|
| `fixed_acidity` | Acidez fixa (g/dm³) |
| `volatile_acidity` | Acidez volátil — alta indica defeito |
| `citric_acid` | Adiciona frescor |
| `residual_sugar` | Açúcar residual após fermentação |
| `chlorides` | Sal — impacta sabor |
| `free_sulfur_dioxide` | Conservante |
| `density` | Correlacionada com álcool e açúcar |
| `pH` | Medida de acidez total |
| `sulphates` | Aditivo antimicrobiano |
| `alcohol` | Teor alcoólico |

---

## O que observar na Feature Importance

`volatile_acidity` e `alcohol` costumam aparecer no topo. Isso faz sentido físico:
- Alta acidez volátil = defeito de fermentação
- Álcool alto = fermentação completa = tendência a qualidade maior

Se uma feature sem lógica física aparecer no topo, investigue antes de confiar.
