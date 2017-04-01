import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# upload data
dataframe = pd.read_csv("input/train-kaggle.csv")

# convert dataframe to matrix

data = dataframe.as_matrix()

unique_label = np.unique(np.array(data[:, 2]))


# function giving a number to a string category

def categorie_numbers(label_list, unique_label):
    converted_label_list = []
    for label in label_list:
        for index in range(len(unique_label)):
            if label == unique_label[index]:
                converted_label_list += [index]
    return np.array(converted_label_list,dtype = int)

# split dataset between train and test

features_train, features_test, label_train, label_test = train_test_split( data[:,3:].astype(np.float64), categorie_numbers(np.array(data[:, 2]),unique_label), test_size=0.33, random_state=42)


# Set up the weights of our perceptron
def set_up_weights(neurone_numbers,unique_label ):
    global synapse_1, biais_1, synapse_2, biais_2
    D = len(features_train[0])
    C = len(np.unique(np.array(unique_label)))
    synapse_1 = 0.001 * np.random.rand(D, neurone_numbers)
    biais_1 = np.zeros(neurone_numbers)
    synapse_2 = 0.001 * np.random.rand(neurone_numbers, C)
    biais_2 = np.zeros(C)


# Neural network definition

def Perceptron(feature_list , label_list = None, loss_function = None,lr = 1e-3, reg=0.0):
    global synapse_1, synapse_2, biais_1, biais_2

    layer_0 = feature_list
    layer_1 = layer_0.dot(synapse_1) + biais_1
    layer_2 = layer_1.dot(synapse_2) + biais_2

    if label_list is None:
        return layer_2

    loss, delta_layer_2 = loss_function(layer_2, label_list)

    delta_synapse_2 = np.dot(layer_1.T, delta_layer_2) + reg * synapse_2
    delta_biais_2 = np.sum(delta_layer_2, axis=0)
    delta_layer_1 = np.dot(delta_layer_2, synapse_2.T)
    delta_biais_1 = np.sum(delta_layer_1, axis=0)
    delta_synapse_1 = np.dot(layer_0.T, delta_layer_1) + reg * synapse_1

    synapse_1 += - lr * delta_synapse_1
    synapse_2 += - lr * delta_synapse_2
    biais_1 += - lr * delta_biais_1
    biais_2 += - lr * delta_biais_2
    return loss

# Softmax will be used as the loss_function

def softmax(label_list_predict, label_list):
    N = label_list_predict.shape[0]
    label_list_predict = label_list_predict.copy()
    label_list_predict -= np.max(label_list_predict, axis=1)[:, None]
    probs = np.exp(label_list_predict)
    probs /= np.sum(probs, axis=1)[:, None]
    loss = np.sum(-np.log(probs[np.arange(N), label_list])) / N

    delta_layer_2 = probs.copy()
    delta_layer_2[np.arange(N), label_list] -= 1

    return loss, delta_layer_2

# Training of the model

num_iterations = 8000
set_up_weights(50 , unique_label)
for i in range(num_iterations):
    loss = Perceptron(features_train, label_train, softmax, lr=0.001, reg=1e-5)
    if i%200 == 0:
        # Accuracy checking
        train_scores = Perceptron(features_train)
        train_acc = (np.argmax(train_scores, axis=1) == label_train).mean()
        val_scores = Perceptron(features_test)
        val_acc = (np.argmax(val_scores, axis=1) == label_test).mean()
        print("number of iterations :",i,"train accuracy :" ,"{0:.0f}%".format(train_acc * 100),"test accuracy :" ,"{0:.0f}%".format(val_acc * 100))

# get true positive:

count_labels_train = Counter(label_train.tolist())
