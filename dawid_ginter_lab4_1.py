import numpy as np
from numpy.core.fromnumeric import transpose
from numpy.core.records import array
import pandas as pd
from sklearn import tree, svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# Zadanie 1 Ładowanie danych treningowych i testotwych
X_test = np.loadtxt("X_test.txt", delimiter=" ")
X_train = np.loadtxt("X_train.txt", delimiter=" ")
y_test = np.loadtxt("y_test.txt", delimiter=" ")
y_train = np.loadtxt("y_train.txt", delimiter=" ")

def prepare_data():  
    global pca_X_train
    global pca_x_test
    pca = PCA()
    pca.fit(X_train)
    pca_X_train = pca.transform(X_train)
    pca_x_test = pca.transform(X_test)

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
    prepare_data()

    # Zadanie 2
    s = svm.SVC(kernel='linear')
    knn = KNeighborsClassifier(4)
    t = tree.DecisionTreeClassifier()
    forest = RandomForestClassifier(max_depth=3, random_state=0)
    score = array([
        cross_val_score(s, pca_X_train, y_train),
        cross_val_score(knn, pca_X_train, y_train),
        cross_val_score(t, pca_X_train, y_train),
        cross_val_score(forest, pca_X_train, y_train)
    ])
    
    print(score)
    df = pd.DataFrame(score)
    df.to_excel(excel_writer="dim_reduction.xlsx")

# Zadanie 3
    random_forest_clf = RandomForestClassifier(n_estimators=10)
    ada = AdaBoostClassifier(n_estimators=10)
    score = array([
        cross_val_score(random_forest_clf, pca_X_train, y_train, cv=5),
        cross_val_score(ada, pca_X_train, y_train, cv=5)
    ])
    df = pd.DataFrame(score)
    df.to_excel(excel_writer="ensambled_learning.xlsx")

if __name__ == "__main__":
    main()
