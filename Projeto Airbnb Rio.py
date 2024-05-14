#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Nosso objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Elas estão disponíveis para download abaixo da aula (se você puxar os dados direto do Kaggle pode ser que encontre resultados diferentes dos meus, afinal as bases de dados podem ter sido atualizadas).
# 
# Caso queira uma outra solução, podemos olhar como referência a solução do usuário Allan Bruno do kaggle no Notebook: https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
# 
# Você vai perceber semelhanças entre a solução que vamos desenvolver aqui e a dele, mas também algumas diferenças significativas no processo de construção do projeto.
# 
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# ### Importar Bibliotecas e Bases de Dados

# In[1]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[2]:


meses = {'jan' : 1, 'fev' : 2, 'mar' : 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul' : 7, 'ago': 8, 'set' : 9 , 'out' : 10, 'nov' : 11, 'dez' : 12}

caminho_bases = pathlib.Path('dataset')

bases = []

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    ano = arquivo.name[-8:]
    ano = ano.replace('.csv', '')
         
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    bases.append(df)

base_airbnb = pd.concat(bases)
display(base_airbnb)


# Vamos fazer uma limpeza nas colunas, o que vai deixar o modelo mais rápido.
# Tipos de colunas a serem excluidas:
# 
# 1- IDs, links e informações não relevantes para o modelo 
# 2- Colunas repetidas ou parecidas com outras que dão a mesma informação para o modelo (ex: data x ano/mês)
# 3- Colunas preenchidas com texto livre (não rodaremos n enhuma análise de palavras)
# 4- Colunas em que todos ou quase todos os valores são iguais
# 
# 
# 
# 
# Para isso, vamos criar um arquivo em excel com os primeiros 1000 registros para uma análise qualitativa.

# In[3]:


#************ NÃO RODAR ESSA CÉLULA ENQUANTO TRABALHA NO CÓDIGO
# print(len(list(base_airbnb.columns)))
# base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')


# #### Depois da análise qualitativa das colunas, levando em conta os critérios explicados acima, ficamos com as seguintes colunas

# In[4]:


arquivo = pd.read_csv(r'primeiros_registros.csv',sep=';')
colunas = list(arquivo.columns)

base_airbnb = base_airbnb.loc[:,colunas]
display(base_airbnb)


# ### Tratar Valores Faltando
# Visualizando os dados percebemos que existe uma grande quantidade de dados faltantes. As colunas com mais de 300.000 valores NaN foram excluidas da análise.
# Para as outras colunas como temos muitos dados (mais de 900.000 linhas) vamos excluir as linhas que contem dados Nan
# 

# In[5]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
        
print(base_airbnb.isnull().sum())
print(base_airbnb.shape)


# In[6]:


base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())
print(base_airbnb.shape)


# ### Verificar Tipos de Dados em cada coluna

# In[7]:


print(base_airbnb.dtypes)
print("-"*60)
print(base_airbnb.iloc[0])


# #### Como as colunas de preço e extra people estão reconhecidas como objeto ao invés de float, vamos mudar o tipo de variável da coluna

# In[8]:


#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)

#extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)

print(base_airbnb.dtypes)


# ### Análise Exploratória e Tratar Outliers
# Vamos olhar feature por feature para:
# 1) ver a correlação entre as features e decidir se manteremos todas as features que temos
# 2) excluir outliers (usaremos como regra valores abaixo de Q1  1.5*Amplitude e valores acima de Q3 +    1.5*Amplitude) Amplitude = Q3 - Q1
# 3) Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma    delas não vai nos ajudar e se devemos excluir.
# - Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também é um valor monetário). Esses são os valores numéricos contínuos.
# - Depois vamos analisar as colunas de valores numéricos discretos (acomodates, guests included, etc)
# - Por fim, vamos avaliar as colunas de texto e decidir quais categorias fazem sentido mantermos ou não.

# In[9]:


plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(numeric_only=True), annot=True, cmap='Greens')
#print(base_airbnb.corr(numeric_only=True))


# ### Definição de funções para análise de outliers

# In[10]:


def limites (coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


# In[11]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)
    
def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# Price

# In[12]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imóveis comuns, acredtio que os valores acima do limite superior serão apenas de imóveis de alto luxo, que não é o o bjetivo do nosso projeto. Por isso, serão excluídos esses outliers.

# In[13]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print('{} linhas removidas'.format(linhas_removidas))


# In[14]:


histograma(base_airbnb['price'])


# Extra people

# In[15]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[16]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print('{} linhas removidas'.format(linhas_removidas))


# In[17]:


histograma(base_airbnb['extra_people'])


# Host listings count

# In[18]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# In[19]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print('{} linhas removidas'.format(linhas_removidas))


# In[20]:


grafico_barra(base_airbnb['host_listings_count'])


# Accomodates

# In[21]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# In[22]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print('{} linhas removidas'.format(linhas_removidas))


# Bathrooms

# In[23]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())


# In[24]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print('{} linhas removidas'.format(linhas_removidas))


# Bedrooms

# In[25]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# In[26]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print('{} linhas removidas'.format(linhas_removidas))


# Beds

# In[27]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# In[28]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print('{} linhas removidas'.format(linhas_removidas))


# Guests included

# In[29]:


diagrama_caixa(base_airbnb['guests_included'])
grafico_barra(base_airbnb['guests_included'])


# A feature "guest included' será excluída da análise.  Aparentemente os usuários do Airbnb usam muito o valor padrão da plataforma como 1 guest included. Isso pode levar o nosso modelo a considerar uma feature que na verdade não é essencial para a definição do preço. Parece melhor excluir a coluna da análise.

# In[30]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape


# Minimum nights

# In[31]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])


# In[32]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print('{} linhas removidas'.format(linhas_removidas))


# Maximum nights

# In[33]:


diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])


# Analisando os dados dessa coluna, ficou claro que também não faz sentido mantê-la na análise, sendo assim, também será excluída

# In[34]:


base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape


# Number of reviews

# In[35]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# A coluna Number of reviews também será excluída da análise

# In[36]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape


# Tratamento de colunas de valores de texto

# Property type

# In[37]:


#print(base_airbnb['property_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'property_type', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)

#property_type


# In[38]:


tabela_tipos_imoveis = base_airbnb['property_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_tipos_imoveis.index:
    if tabela_tipos_imoveis[tipo] < 2000:
        colunas_agrupar.append(tipo)
#print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'

print(base_airbnb['property_type'].value_counts().index)
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'property_type', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# Room type

# In[39]:


print(base_airbnb['room_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'room_type', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# Bed type

# In[40]:


print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'bed_type', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)

#agrupando categorias de bed type

tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)
#print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'

print(base_airbnb['bed_type'].value_counts().index)
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'bed_type', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# Cancellation policy

# In[41]:


print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'cancellation_policy', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# agrupando categorias de cancellation policy

tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)
#print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'Strict'

print(base_airbnb['cancellation_policy'].value_counts().index)
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'cancellation_policy', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# Room type

# In[42]:


print(base_airbnb['room_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot(x = 'room_type', data = base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# Amenities
# ****
# Devido a dificuldade de tratameno dos dados dessa coluna, inclusive pela sua diversidade e falta de padronização, ficou inviável a sua análise. Sendo assim, iremos contar a quantidade de amenities de cada imóvel e usar isso como parâmetro, acreditando que quanto mais amenities, significa mais cara esse imóvel será, inclusive pela atenção e dedicação do host em descrevê-los.

# In[43]:


print(base_airbnb['amenities'].iloc[0].split(','))
print(len(base_airbnb['amenities'].iloc[0].split(',')))

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)


# In[44]:


base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape


# In[45]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# In[46]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print('{} linhas removidas'.format(linhas_removidas))


# ### Visualizaçao de Mapa das Propriedades

# In[47]:


amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='open-street-map')
mapa.show()


# ### Encoding
# *Precisamos ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true or false, etc).
# *Fetarures de categoria (features em que ps vaçpres da coluna são textos) vamos utilizar o método de encoding de variáveis dummies

# In[48]:


colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready' ]
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f', coluna] = 0
#print(base_airbnb_cod.iloc[0])


# In[49]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy' ]
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias, dtype=int)
display(base_airbnb_cod.head())


# ### Modelo de Previsão

# Métricas de avaliação

# In[50]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nr²: {r2:.2%}\nRSME:{RSME:.2f}'


# Escolha dos modelos a serem testados:<br>
# 1. RandomForest<br>
# 2. LinearRegression<br>
# 3. ExtraTree

# In[51]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegressor': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)


# In[52]:


display(X)


# * Separar os dados em treino e teste e depois Treinar o modelo

# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# ### Análise do Melhor Modelo

# In[54]:


for nome_modelo, modelo in modelos.items():
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# Modelo escolhido como melhor modelo: ExtraTreeRegressor
# 
# Esse foi o modelo com o maior R2 e ao mesmo tempo o menor valor de RSME. Como não tivemos uma grande diferença de velocidade e de previsão desse modelo com o modelo de RandomForest (que teve resultados próximos de R2 e RSME), vamos escolher o modelo ExtraTrees.
# 
# O modelo de regressão linear não teve resultado satisfatório, com valores de R2 e RSME muito piores do que os outros dois modelos
# 
# Resultados das métricas de avaliação do modelo vencedor:<br>
# Modelo ExtraTrees:<br>
# r²: 97.52%<br>
# RSME:41.76

# ### Ajustes e Melhorias no Melhor Modelo

# In[55]:


importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)

plt.figure(figsize=(15,5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)


# #### Ajustes finais no modelo
# is_business_travel_ready não parece ter feito impacto no modelo. Por isso, para chegar em um modelo mais simples, vamos excluir essa feature para efetuar novos testes

# In[56]:


base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
modelo.fit(X_train, y_train)
previsao = modelo.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# Excluindo a coluna 'is_business_travel_ready', o modelo teve praticamente o mesmo resultado, porém ficou um pouco mais simples, o que também o deixa um pouco mais rápido.

# In[61]:


base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)

y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
modelo.fit(X_train, y_train)
previsao = modelo.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# In[63]:


print(previsao)


# # Deploy do Modelo
# 

# In[64]:


X['price'] = y
X.to_csv('dados.csv')


# In[65]:


import joblib
joblib.dump(modelo_et, 'modelo.joblib')


# In[ ]:




