# Módulo 08 — Mentalidade

## O que separa quem aprende ML de quem usa ML

Existe uma diferença entre quem executa código e quem resolve problemas com ML.

Executar código é fácil. Qualquer pessoa com acesso ao Stack Overflow faz isso.

Resolver problemas exige entender o que está acontecendo.

---

## As perguntas certas

Antes de treinar qualquer modelo, responda:

**1. Qual é a decisão que esse modelo vai suportar?**
ML não é um fim. É uma ferramenta para tomar decisões melhores, mais rápido.

**2. O que é um erro grave aqui?**
Falso positivo (bloquear cliente legítimo) ou falso negativo (fraude passar)?
A resposta define a métrica e o threshold.

**3. Existe data leakage?**
Use apenas informações disponíveis no momento da previsão.

**4. Por que essa feature faz sentido?**
Nunca crie feature sem hipótese. Feature sem hipótese é ruído com nome bonito.

---

## Os erros mais comuns (e como evitar)

**Otimizar acurácia em dataset desbalanceado**
→ Use ROC-AUC ou PR-AUC. Acurácia alta com modelo inútil é o erro mais fácil de cometer.

**Avaliar o modelo no treino**
→ Sempre `X_test`. Modelo que só funciona no treino não serve para nada.

**Achar que mais features = modelo melhor**
→ Features com ruído degradam o modelo. Qualidade > quantidade.

**Pular validação cruzada**
→ Um split pode ter sorte. Cinco splits mostram a realidade.

**Considerar o notebook como entrega final**
→ Modelo em notebook não está em produção. MLflow é o primeiro passo.

---

## A progressão real

```
Nível 1 → Rodar código copiado
Nível 2 → Entender o que cada linha faz
Nível 3 → Saber por que escolheu aquela abordagem
Nível 4 → Adaptar para um problema novo sem tutorial
Nível 5 → Identificar o que pode dar errado antes de dar
```

Este curso cobre os níveis 2 e 3. Os próximos níveis vêm com prática em problemas reais.

---

## O próximo passo

Pegue um dataset que você não criou, com um problema que importa para alguém, e construa um modelo do zero — sem olhar para o código deste curso.

Se conseguir, você aprendeu.
