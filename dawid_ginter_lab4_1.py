import numpy as np
from numpy.core.records import array
import pandas as pd
from sklearn import tree, svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Zadanie 1 Ładowanie danych treningowych i testotwych
X_test = np.loadtxt("X_test.txt", delimiter=" ")
X_train = np.loadtxt("X_train.txt", delimiter=" ")
y_test = np.loadtxt("y_test.txt", delimiter=" ")
y_train = np.loadtxt("y_train.txt", delimiter=" ")


pca = PCA()
pca.fit(X_train)

# def svmClassifier():
#     svm.SVC(kernel='linear')
#     # clf = clf.fit(X_train, y_train)
#     # clf.predict(X_test)

# def knnClassifier(n_neighbors: int):
#     knn = KNeighborsClassifier(n_neighbors)
#     # knn.fit(X_train, y_train)
#     # y_pred = knn.predict(X_test)

# def decisionTreeClsf():
#     clf = tree.DecisionTreeClassifier()
#     # clf = clf.fit(X_train, y_train)

# def randomForestClsf(max_depth, random_state):
#     clf = RandomForestClassifier(max_depth, random_state)
#     # clf.fit(X_train, y_train)


def main():
    score = []
    s = svm.SVC(kernel='linear')
    knn = KNeighborsClassifier(4)
    t = tree.DecisionTreeClassifier()
    forest = RandomForestClassifier(max_depth=3, random_state=0)
    score = array([
        cross_val_score(s, X_train, y_train),
        cross_val_score(knn, X_train, y_train),
        cross_val_score(t, X_train, y_train),
        cross_val_score(forest, X_train, y_train)
    ])
    # score[0] = cross_val_score(s, X_train, y_train)
    # score[1] = cross_val_score(knn, X_train, y_train)
    # score[2] = cross_val_score(t, X_train, y_train)
    # score[3] = cross_val_score(forest, X_train, y_train)
    
    print(score)
    df = pd.DataFrame(score)
    df.to_excel(excel_writer="dim_reduction.xlsx")

if __name__ == "__main__":
    main()