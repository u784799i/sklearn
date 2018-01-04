import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target

##print(iris_X)
##print(iris_y)

X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)

knn=KNeighborsClassifier()
#knn.fit(X_train,y_train)
#print(knn.predict(X_test))
#print(y_test)
knn.fit(iris_X,iris_y)
#print(knn.score(X_test,y_test))

scores=cross_val_score(knn,iris_X,iris_y,cv=5,scoring='accuracy')
print(scores.mean())


print(knn.get_params())
