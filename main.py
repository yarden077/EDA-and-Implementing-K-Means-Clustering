from data import load_data, add_new_columns, data_analysis
from clustering import transform_data, kmeans, visualize_results
import numpy as np
import sys


def main(argv):
    path = '/Users/yardennahum/Desktop/HW2/london_sample_500.csv'
    df = load_data(path)
    df = add_new_columns(df)
    data_analysis(df)
    features = ['cnt', 'hum']
    df = transform_data(df, features)
    print()
    print("Part B:")
    k = [2, 3, 5]
    path = '/Users/yardennahum/Desktop/HW2/plot{}.png'
    for i in k:
        print(f"k = {i}")
        labels, centroids = kmeans(df, i)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        visualize_results(df, labels, centroids, path)
        if i == 5:
            break
        print()


if __name__ == '__main__':
    main(sys.argv)
