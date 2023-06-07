import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def svm_algorithm():
    cancer = datasets.load_breast_cancer()

    X = cancer.data
    y = cancer.target

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    classes = ["malignant", "benign"]

    # generate the support vector
    clf = svm.SVC(kernel= "linear", C = 2)
    clf.fit(x_train, y_train)


    y_pred = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("svc:", acc)

    # compare with KNearest neighbors
    KClf = KNeighborsClassifier(n_neighbors=13)
    KClf.fit(x_train, y_train)
    y_pred = KClf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("KClf: ", acc)

