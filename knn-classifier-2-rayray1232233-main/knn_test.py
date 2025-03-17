import pandas as pd
import numpy as np
from knn_classifier import knn_predict, knn_calc_dists

# Test if KNN accuracy on Iris flower dataset is higher than random-guess (> 0.33).

def test_knn_classifier_accuracy():
    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv("./data/IRIS.csv")
    train_data = train_df.iloc[:,:-1].to_numpy()
    train_label = train_df.iloc[:,-1:].to_numpy() # Split labels in last column

    test_df = pd.read_csv("./data/iris_test.csv")
    test_data = test_df.iloc[:,:-1].to_numpy()
    test_label = test_df.iloc[:,-1:].to_numpy() # Split labels in last column

    K = 3
    predictions = knn_predict(train_data, train_label, test_data, K)

    # Calculate accuracy
    result = predictions == test_label
    accuracy = sum(result == True) / len(result)
    print('Evaluate KNN(K=%d) on Iris Flower dataset. Accuracy = %.2f' % (K, accuracy))

    assert accuracy > 0.33


# Test if matrix calculation is correct

def test_knn_distances():
    train_data = np.random.rand(100, 4)
    test_data = np.random.rand(23, 4)
    K = 5
    
    indices, distances = knn_calc_dists(train_data, test_data, K)
    
    assert np.sum(distances) != 0
    assert sum(distances.shape) == len(test_data) + K
