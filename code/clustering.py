from data import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe.
    :return: transformed data as numpy array of shape (n, 2)
    """
    df = load_data('london_sample_500.csv')
    number_of_rows = df.shape[0]
    transformed_data = df[features].to_numpy()
    transformed_data = sum_scaling(transformed_data, number_of_rows)
    transformed_data = add_noise(transformed_data)
    return transformed_data


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    .:param data: numpy array of shape (n, 2)
    :param data:
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    not_finished = True
    labels = np.zeros(shape=data.shape[0])
    current_centroids = choose_initial_centroids(data, k)
    while not_finished:
        labels = assign_to_clusters(data, current_centroids)
        prev_centroids = current_centroids
        current_centroids = recompute_centroids(data, labels, k)
        if np.array_equal(prev_centroids, current_centroids):
            not_finished = False
    return labels, current_centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    .:param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    features = ['cnt', 'hum']
    colors = np.array(['purple', 'green', 'blue', 'red', 'yellow'])
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]
    for i in range(centroids.shape[0]):
        plt.scatter(centroids_x, centroids_y, color='white', edgecolors='black', marker='*', linewidth=3,
                    s=600, alpha=0.85, label=f'Centroid' if i == 0 else None)
    plt.xlabel(f'{features[0]}')
    plt.ylabel(f'{features[1]}')
    plt.title(f'Results for kmeans with k = {centroids.shape[0]}')
    plt.legend()
    plt.savefig(path.format(centroids.shape[0]))


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    distance = np.linalg.norm(x - y)
    return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    number_of_samples = data.shape[0]
    labels = np.zeros(shape=number_of_samples)
    number_of_clusters = centroids.shape[0]
    for i in range(number_of_samples):
        min_distance = 0
        for j in range(number_of_clusters):
            distance = dist(data[i], centroids[j])
            if distance < min_distance or min_distance == 0:
                min_distance = distance
                labels[i] = j
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids = np.zeros(shape=(k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(data[np.where(labels == i)], axis=0)
    return centroids


def sum_scaling(data, number_of_rows):
    """ This function scales every value in our data by sum scaling """
    cnt = data[:, 0]
    hum = data[:, 1]
    min_cnt = cnt.min()
    min_hum = hum.min()
    sum_cnt = cnt.sum()
    sum_hum = hum.sum()
    for i in range(number_of_rows):
        cnt[i] = (cnt[i] - min_cnt) / sum_cnt
        hum[i] = (hum[i] - min_hum) / sum_hum
    return data
