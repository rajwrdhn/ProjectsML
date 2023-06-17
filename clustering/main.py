
# agglomerative clustering
import pandas as pd
from numpy import unique
from numpy import where
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot

# read the dataset 
df = pd.read_csv("data/online_shoppers_intention.csv")
print(df.columns)
#no nan values
#df.dropna(axis=1)
#drop categorical columns
df = df.drop(columns=['Month','VisitorType','Weekend', 'Revenue'])
print(df)


# define dataset
data = df.values

# define the model
model = AgglomerativeClustering(n_clusters=30)
# fit model and predict clusters
yhat = model.fit_predict(data)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(data[row_ix, 0], data[row_ix, 1])
# show the plot
pyplot.show()
