# Módulo 00 — Setup Databricks

## Pré-requisitos

- Conta Google ou Microsoft (para login)
- Navegador atualizado
- Nenhuma instalação local necessária

---

## Passo 1 — Criar conta no Databricks Free Edition

1. Acesse [databricks.com/try-databricks](https://www.databricks.com/try-databricks)
2. Selecione **Community Edition** (gratuita, sem cartão)
3. Preencha nome, e-mail e senha
4. Confirme o e-mail

---

## Passo 2 — Criar um Cluster

O cluster é o ambiente de execução. Sem ele, nenhum notebook roda.

1. No menu lateral, clique em **Compute**
2. Clique em **Create Compute**
3. Mantenha as configurações padrão
4. Clique em **Create Compute**
5. Aguarde o status mudar para **Running** (1–3 minutos)

> A Community Edition desliga o cluster automaticamente após 2h de inatividade. Basta reiniciar quando precisar.

---

## Passo 3 — Fazer upload dos datasets

1. No menu lateral, clique em **Data** → **Add Data**
2. Selecione **Upload File**
3. Faça upload dos 4 arquivos da pasta `datasets/`:
   - `churn.csv`
   - `fraud.csv`
   - `logistics.csv`
   - `quality.csv`
4. Os arquivos ficam disponíveis em `/dbfs/FileStore/datasets/`

---

## Passo 4 — Criar um Notebook

1. No menu lateral, clique em **New** → **Notebook**
2. Escolha o nome (ex: `01_base_model`)
3. Linguagem: **Python**
4. Selecione o cluster criado no Passo 2

---

## Passo 5 — Importar os notebooks do curso

1. No menu lateral, clique em **Workspace**
2. Clique em **Import**
3. Selecione os arquivos `.py` da pasta `notebooks/`
4. Execute célula por célula com `Shift + Enter`

---

## Verificar se está tudo funcionando

Cole isso em uma célula e execute:

```python
import pandas as pd
df = pd.read_csv("/dbfs/FileStore/datasets/churn.csv")
print(df.shape)
display(df.head())
```

Se retornar `(2000, 7)` e mostrar a tabela, está tudo certo.

---

## Dica

Não pule etapas.  
Machine Learning é execução — ambiente quebrado = aprendizado zero.
