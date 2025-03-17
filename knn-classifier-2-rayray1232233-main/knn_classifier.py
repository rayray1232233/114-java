# -*- coding: utf-8 -*-
# Reference: https://nycdatascience.com/blog/student-works/machine-learning/knn-classifier-from-scratch-numpy-only/

import pandas as pd
import numpy as np
import argparse

KNN = 2
TRAIN_FILE = 'IRIS.csv'         # 訓練集:像有解答的題目
TEST_FILE = 'iris_test.csv'     #測試及:沒有解答代表你真的會

# Calculate all Euclidean distances between training and test data

def knn_calc_dists(xTrain, xTest, k):
    """
    Finds the k nearest neighbors of xTest in xTrain.
    Input:
        xTrain = n x d matrix. n=rows and d=features
        xTest = m x d matrix. m=rows and d=features (same amount of features as xTrain)
        k = number of nearest neighbors to be found
    Output:
        dists = distances between xTrain/xTest points. Size of n x m 
        indices = kxm matrix with indices of yTrain labels
    """
    distances = -2 * np.dot(xTrain, xTest.T)
    distances += np.sum(xTest**2, axis=1)
    distances += np.sum(xTrain**2, axis=1)[:, np.newaxis]
    distances = np.sqrt(distances)
    indices = np.argsort(distances,axis=0) #距離排序用索引呈現 
    distances = np.sort(distances,axis=0)  #直接排
    
    return indices[0:k, : ], distances  #選0到k-1行  : 表全部得列 後面遠得都捨棄

def knn_predict(xTrain, yTrain, xTest, k=3): #近朱者赤找最近作為標籤 xtrain:訓練資料ex:蘋果 ytrain:水果名稱標籤 xtest:想知道的新水果  1.排序距離 
    """
    Input:
        xTrain = n x d matrix. n=rows and d=features
        yTrain = n x 1 array. n=rows with label value
        xTest = m x d matrix. m=rows and d=features
        k = number of nearest neighbors to be found
    Output:
        predictions = predicted labels, ie preds(i) is the predicted label of xTest(i,:)
    """
    indices, distances = knn_calc_dists(xTrain, xTest, k) #取dist值
    rows, columns = indices.shape                         #取array形狀
    predictions = []
    for j in range(columns):
        k_nearest_indices=indices[:, j]                       #取第j列全部行
        k_nearest_labels = yTrain[k_nearest_indices]       #k最近idx數列從標籤列中取出
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True) #np.unique每個值出現的次數
        #ex.K個最近標籤 ['蘋果', '香蕉', '蘋果', '蘋果', '橘子'] unique label=['蘋果', '橘子', '香蕉'] count=[3, 1, 1] 
        most_idx=np.argmax(counts)
        most_label=unique_labels[most_idx]

        predictions.append(most_label)
    
    return np.array(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K Nearest Neighbor Classifier")
    parser.add_argument("--train-csv", help="Training data in CSV format. Labels are stored in the last column.", required=True)
    parser.add_argument("--test-csv", help="Test data in CSV format", required=True)
    parser.add_argument("--num_k", "-k", dest="K", help="Number of nearest neighbors", default=3, type=int)
    args = parser.parse_args()

    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv(args.train_csv)
    train_data = train_df.iloc[:,:-1].to_numpy()
    train_label = train_df.iloc[:,-1:].to_numpy() # Split labels in last column

    test_df = pd.read_csv(args.test_csv)
    test_data = test_df.iloc[:,:-1].to_numpy()
    test_label = test_df.iloc[:,-1:].to_numpy() # Split labels in last column

    predictions = knn_predict(train_data, train_label, test_data, args.K)

    # Save prediction results
    #np.savetxt("predictions.csv", predictions, delimiter=',') # Not working for strings
    df = pd.DataFrame(predictions)
    df.to_csv("predictions.csv", header=False, index=False)

    # Calculate accuracy
    result = predictions == test_label
    accuracy = sum(result == True) / len(result)
    print('Evaluate KNN(K=%d) on Iris Flower dataset. Accuracy = %.2f' % (args.K, accuracy))
