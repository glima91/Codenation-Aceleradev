#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[23]:


black_friday.head(5)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[15]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[16]:


def q2():
    selected = black_friday[(black_friday.Gender == 'F') & (black_friday.Age == '26-35')]
    qnt = selected.shape[0]
    return qnt
    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[17]:


def q3():
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[18]:


def q4():
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[20]:


def q5():
    n_na = black_friday[black_friday.isnull().any(axis=1)]
    return n_na.shape[0]/black_friday.shape[0]


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[20]:


def q6():
    n_na = black_friday.isnull().sum()
    return max(n_na)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[21]:


def q7():
    return black_friday['Product_Category_3'].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[22]:


def q8():
    min_val = black_friday['Purchase'].min()
    max_val = black_friday['Purchase'].max()
    df_norm = (black_friday['Purchase'] - min_val)/(max_val - min_val)
    df_norm_mean = df_norm.mean()
    # converte de numpy.float64 para tipo float
    df_norm_mean = df_norm_mean.item()
    return df_norm_mean


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[23]:


def q9():
    df_norm = (black_friday['Purchase'] - black_friday['Purchase'].mean())/black_friday['Purchase'].std()
    num_ocorrencias = df_norm[(df_norm <=1 ) & (df_norm >-1)].count()
    # converte do formato numpy.int32 para int
    num_ocorrencias = num_ocorrencias.item()
    return num_ocorrencias


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[24]:


def q10():
    x1 = black_friday[(black_friday['Product_Category_2'].isnull()) & (black_friday['Product_Category_3'].notnull())]
    if x1.shape[0] == 0:
        return True
    else:
        return False

