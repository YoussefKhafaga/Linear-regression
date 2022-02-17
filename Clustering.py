import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.cluster import KMeans


def read_data():
    return pd.read_csv("Iris.csv")


if __name__ == '__main__':

    #data read
    data = read_data()
    x = data.iloc[:, [0, 1, 2, 3]].values
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    # Plotting the results onto a line graph,
    # `allowing us to observe 'The elbow'
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # Within cluster sum of squares
    plt.show()

    #From the elbow we take n_clusters as 3
    kmeans = KMeans(n_clusters=3, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(x)

    # Visualising the clusters - On the first two columns
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],
                s=100, c='red', label='Iris-setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
                s=100, c='blue', label='Iris-versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
                s=100, c='green', label='Iris-virginica')

    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=100, c='yellow', label='Centroids')

    plt.legend()
    plt.show()