import numpy as np
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Zadanie 1 Ładowanie danych treningowych i testotwych
X_test = np.loadtxt("X_test.txt", delimiter=" ")
X_train = np.loadtxt("X_train.txt", delimiter=" ")
y_test = np.loadtxt("y_test.txt", delimiter=" ")
y_train = np.loadtxt("y_train.txt", delimiter=" ")


pca = PCA()
pca.fit(X_train)

def knnClassifier(n_neighbors: int):
    scores_list = []
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

def decisionTreeClsf():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

def randomForestClsf(max_depth, random_state):
    clf = RandomForestClassifier(max_depth, random_state)
    clf.fit(X_train, y_train)