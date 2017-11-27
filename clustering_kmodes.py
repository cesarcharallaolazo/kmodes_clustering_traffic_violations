"""

@Author: Cesar Augusto Charalla Olazo
@Objective: Clustering of traffic violations in Miraflores district - Lima - Peru
@Technical Objective: Know how to use k-modes algorithm in clustering models in python

"""

# import libraries
import pandas as pd
from kmodes.kmodes import KModes

# get excel book in pandas
xls_file = pd.ExcelFile("source/traffic_violations_selected_features_delete_missing.xlsx")

# get excel sheet - object type: pandas dataframe
pd_traffic_violations = xls_file.parse('Hoja1')

# select features to model
pd_traffic_violations_to_model = pd_traffic_violations[['CODIGO INFRACCION','TIPO DE VIA', 'LUGAR DE INTERVENCION','EMPRESA DE TRANSPORTE']]

# instance k-modes object - 180 clusters
kmodes = KModes(n_clusters = 180 , init='Cao', verbose=1)
# modeling
kmodes.fit(pd_traffic_violations_to_model)

# cluster centroids of the model
print(kmodes.cluster_centroids_)
# statistics of modeling
print(kmodes.cost_)
print(kmodes.n_iter_)

# create new cluster column in pandas dataframe
pd_traffic_violations['CLUSTER']  = kmodes.labels_ 

# save labeled dataframe to .csv
pd_traffic_violations.to_csv('clustering/kmodes_clustering_traffic_violations.csv', index = False , header = True)

