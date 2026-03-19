# Módulo 03 — Churn

## Contexto de negócio

Uma empresa de telecomunicações quer saber **quais clientes vão cancelar o serviço nos próximos 30 dias**.

Com isso, o time de retenção pode agir preventivamente: oferecer desconto, ligar, mudar o plano.

Sem o modelo, o time age às cegas ou espera o cancelamento acontecer.

---

## O que você vai aprender

- Feature Engineering com hipótese de negócio
- Por que `tenure` é a feature mais importante em churn
- Como interpretar Precision e Recall no contexto de retenção
- Curva ROC e o que ela representa

---

## Features do dataset

| Feature | Tipo | Significado |
|---|---|---|
| `tenure` | numérica | Meses como cliente |
| `monthly_charge` | numérica | Valor mensal pago |
| `support_calls` | numérica | Chamadas para suporte |
| `num_products` | numérica | Produtos contratados |
| `has_contract` | binária | Tem contrato? |
| `late_payments` | numérica | Pagamentos em atraso |
| `churn` | target | 1 = cancelou |

---

## Feature Engineering

Criamos duas features derivadas com hipóteses claras:

**`charge_per_month = monthly_charge / (tenure + 1)`**

Clientes novos pagando muito têm mais chance de cancelar do que clientes antigos pagando o mesmo valor. O custo-benefício percebido é diferente.

**`calls_per_product = support_calls / (num_products + 1)`**

Muito suporte para poucos produtos sinaliza frustração com qualidade — não apenas uso intenso.

---

## Métricas para churn

**Precision** → dos clientes que prevemos que vão cancelar, quantos realmente cancelaram?
→ baixa precision = equipe de retenção liga para clientes que ficariam de qualquer forma (custo desnecessário)

**Recall** → dos clientes que realmente foram cancelar, quantos identificamos?
→ baixo recall = perdemos clientes que poderiam ter sido retidos (receita perdida)

O trade-off depende do custo de cada erro para o negócio.

---

## Erros comuns neste módulo

**Data leakage:** usar informações que só existem após o cancelamento como features. Exemplo: `dias_desde_ultimo_uso` medido depois que o cliente já cancelou.

**Classe desbalanceada leve:** churn em torno de 20-30% ainda permite `accuracy` como métrica, mas prefira ROC-AUC para avaliação.
