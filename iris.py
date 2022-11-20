import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data processing
data_iris = pd.read_csv('Iris.csv')
data_iris = data_iris.drop(['Id'], axis = 1)

y_iris = data_iris['Species']
data_iris = data_iris.drop(['Species'], axis = 1)


x_train, x_test, y_train, y_test = train_test_split(data_iris, y_iris, test_size = 0.2, random_state = 27)

# use several machine learning models
# and find the deviation value for each
knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(x_train, y_train)
knn_prediction = knn_model.predict(x_test)
print('KNN - ', accuracy_score(knn_prediction, y_test))

svc_model = SVC()
svc_model.fit(x_train, y_train)
svc_prediction = svc_model.predict(x_test)
print('SVC - ', accuracy_score(svc_prediction, y_test))
print()


variety = {'Iris-virginica':1, 'Iris-setosa':2, 'Iris-versicolor':3}
svc_prediction = [variety[items] for items in svc_prediction]
knn_prediction = [variety[items] for items in knn_prediction]
y_test = [variety[items] for items in y_test]

print(svc_prediction)
print(knn_prediction)
print(y_test)


# model response graphs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot([i for i in range(len(y_test))], y_test, 'c.-', [i for i in range(len(y_test))], svc_prediction, 'r.', [i for i in range(len(y_test))], knn_prediction, 'k.')
plt.axis([0, 30, 0, 4])
plt.show()