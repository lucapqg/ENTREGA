# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
plt.style.use('ggplot')

# COMMAND ----------

df_snis = pd.read_csv("/dbfs/FileStore/tables/Dados_SNIS_datathon.csv", header='infer', error_bad_lines=False,sep=',',decimal=',')
df_snis.head()

# COMMAND ----------

df_snis.info()

# COMMAND ----------

pd.DataFrame(df_snis.isna().sum(),columns=['Quantidade de nulos'])

# COMMAND ----------

df_snis.describe().T

# COMMAND ----------

corr = pd.DataFrame(df_snis.corr('pearson')['IN049 - Índice de perdas na distribuição']).drop(['Código do Município'],axis=0)
cor_perdas = corr[(corr['IN049 - Índice de perdas na distribuição']>0.2)|(corr['IN049 - Índice de perdas na distribuição']<-0.2)].sort_values(by='IN049 - Índice de perdas na distribuição', ascending=False)
cor_perdas

# COMMAND ----------

cor_perdas

# COMMAND ----------

plt.subplots(figsize=(25,7))
ax = sns.barplot(y=cor_perdas.index,x=cor_perdas['IN049 - Índice de perdas na distribuição'])
ax.set_title("Correlação de pearson de índice de perdas na distribuição x demais índices de eficiência operacional",fontsize=20)
ax.tick_params(labelsize=19)
plt.show(ax)

# COMMAND ----------

plt.subplots(figsize=(25,7))
sns.heatmap(cor_perdas,linewidths=.5)

# COMMAND ----------

df_snis[cor_perdas.index].sort_values('IN049 - Índice de perdas na distribuição',ascending=False).iloc[0:10]

# COMMAND ----------

df_snis[cor_perdas.index].sort_values('IN049 - Índice de perdas na distribuição',ascending=False).iloc[-10:]

# COMMAND ----------

cols = corr[(corr['IN049 - Índice de perdas na distribuição']>0.2)|(corr['IN049 - Índice de perdas na distribuição']<-0.2)].index

g = sns.pairplot(df_snis[cols])
for ax in g.axes.flatten():
    # rotate x axis labels
    ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    # rotate y axis labels
    ax.set_ylabel(ax.get_ylabel(), rotation = 0)
    # set y labels alignment
    ax.yaxis.get_label().set_horizontalalignment('right')

plt.show(g) 

# COMMAND ----------

df_num = df_snis.loc[:,cor_perdas.index].dropna(axis=0)

# COMMAND ----------

scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(df_num))
scaled.columns = cor_perdas.index

scaled.head()

# COMMAND ----------

# instanciando o modelo
kmeans = KMeans(n_clusters =2, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(scaled)

# COMMAND ----------

# visualizando os clusters
fig,_=plt.subplots(figsize=(13,11))
scaled = scaler.fit_transform(df_num)
# definindo a figura em 3 dimensões
ax = fig.add_subplot(111, projection='3d')

# ax.azim = 10
# ax.dist = 10
ax.elev = 70


# plotando os clusters
plt.scatter(scaled[y_kmeans == 0,0],scaled[y_kmeans == 0,1],  s= 50, c= 'red',label= 'Cluster 1')
plt.scatter(scaled[y_kmeans == 1,0], scaled[y_kmeans == 1,1], s= 50, c= 'blue', label= 'Cluster 2')
# plt.scatter(scaled[y_kmeans == 2,0], scaled[y_kmeans == 2,1], s= 50, c= 'green', label= 'Cluster 3')
# plt.scatter(scaled[y_kmeans == 3,0], scaled[y_kmeans == 3,1], s= 50, c= 'cyan', label= 'Cluster 4')
# plt.scatter(scaled[y_kmeans == 4,0], scaled[y_kmeans == 4,1], s= 50, c= 'magenta', label= 'Cluster 5')
# plt.scatter(scaled[y_kmeans == 5,0], scaled[y_kmeans == 5,1], s= 50, c= 'gray', label= 'Cluster 6')


# centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s= 300, c= 'yellow', label= 'Centroids')
plt.title('Clusters',fontsize=20,fontweight='bold')
plt.legend()
plt.show()
