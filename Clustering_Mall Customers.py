import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df=pd.read_csv(r'C:\Users\Vatsal\Desktop\projects\Data Sets\Class\Mall_Customers.csv')
df.head()
#%%
df.shape
df.isnull().sum()
X = df.values[:,[3,4]]   ## conervting a data frame object into array, as it runs faster compared to dataframe
X
#%%
## Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wsse = []   ## wsse = within the clusters sum of sqaured errors
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 10) ## random_state refers to set.seed i.e., random grouping
    kmeans.fit(X)  ## fit refers to training data
    wsse.append(kmeans.inertia_) ## Finding the error value for wsse using inertia function
plt.plot(range(1, 11), wsse)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WSSE')
plt.show()  ## From the graph we conclude that point 5 is the optimum point for K from where the error starts declining
#%%
kmeans = KMeans(n_clusters=5, random_state = 10)
Y_pred = kmeans.fit_predict(X)
####### MODEL BUILDING
## 1. Create the model object where you call the algorithm function
## 2. Train the model object where you call the fit() function
## 3. Use the model for prediction where you call the predict() function
Y_pred
#%%
###### Visualising the clusters
#plt.scatter(X,Y)
#X==>0th col(Annual Income)
#Y==>1st col(Spending Score)
#X[Y_pred == 0, 0]==>X[all obs which have been assigned to cluster 0,Annual income variable]
#X[Y_pred == 2, 1]==>X[all obs which have been assigned to cluster 2,Spending score variable]
plt.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster1')
plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster2')
plt.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster3')
plt.scatter(X[Y_pred == 3, 0], X[Y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster4')
plt.scatter(X[Y_pred == 4, 0], X[Y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#%%
# Interpretation: 
#Green- Target customers
#Red -  Second target
#%%
plt.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s = 50, c = 'red', label = 'Careless')
plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s = 50, c = 'green', label = 'Target')
plt.scatter(X[Y_pred == 3, 0], X[Y_pred == 3, 1], s = 50, c = 'cyan', label = 'Sensible')
plt.scatter(X[Y_pred == 4, 0], X[Y_pred == 4, 1], s = 50, c = 'magenta', label = 'Careful')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#%%
df['Cluster'] = Y_pred
df.head()
df["Cluster"]=df.Cluster.map({0:'Careless',1:'Standard',2:'Target',3:'Sensible',4:'Careful'})
df.head()
#%%
new_df=df[df["Cluster"]=="Target"]
new_df.shape
new_df.to_excel("TargetCustomers.xlsx", index=False)
df.to_excel('Mall_Customers Cluster.xlsx', index = False)
