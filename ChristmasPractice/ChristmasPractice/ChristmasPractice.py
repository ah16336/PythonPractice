import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def rate_score(classy, x_test, y_test):
    score = classy.score(x_test, y_test)
    score = round(score, 3)
    print("Score:", score)
    if score > 0.9:
        print("That's a fantastic score!")
    elif score > 0.8:
        print("That's a great score.")
    elif score > 0.7:
        print("That's a good score.")
    elif score > 0.5:
        print("That's an OK score but could be improved.")
    else:
        print("This score is not so good. We need to rethink our model.")

iris_df = sns.load_dataset('iris')

## LINEAR REGRESSION ##
print("-- SIMPLE LINEAR REGRESSION -- \n")

x = iris_df[["sepal_length"]]
y = iris_df[["petal_length"]]

#plt.scatter(x, y)
#plt.show()

x = x.values.reshape(-1, 1)
line_x_train, line_x_test, line_y_train, line_y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

from sklearn.linear_model import LinearRegression

line = LinearRegression()
line.fit(line_x_train, line_y_train)

print("Line Equation: y =", line.coef_, "x +", line.intercept_)

rate_score(line, line_x_test, line_y_test)

unknowns = np.array([8.2, 5.7, 4.5])
unknowns = unknowns.reshape(-1, 1)

predictions = line.predict(unknowns)

plt.scatter(x, y, label='Data Points')
plt.plot(x, x*line.coef_+line.intercept_, color='Orange', label='Line Eqn. Calculated')
plt.scatter(unknowns, predictions, color='Red', label='Predictions', marker='s')
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Relationship between Petal and Sepal Length with Predictions")
plt.legend()
plt.show()

## MULTIPLE LINEAR REGRESSION ##
print("\n -- MULTIPLE LINEAR REGRESSION -- \n")

features = iris_df[['sepal_length', 'sepal_width', 'petal_width']]
# y is unchanged

mlr_x_train, mlr_x_test, mlr_y_train, mlr_y_test = train_test_split(features, y, train_size=0.8, test_size=0.2)
mlr = LinearRegression()
mlr.fit(mlr_x_train, mlr_y_train)

# Which variable had the biggest impact?
print("Features: [Sepal Length, Sepal Width, Petal Width]")
print("Coefficients:", mlr.coef_)

rate_score(mlr, mlr_x_test, mlr_y_test)

x_predict = [[4.6, 3.2, 0.2]]
y_predict = mlr.predict(x_predict)

print("Prediction. Features:", x_predict, "Predicted petal length:", y_predict)


## K NEAREST NEIGHBOURS CLASSIFICATION ##
print("\n -- K NEAREST NEIGHBOURS CLASSIFICATION -- \n")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score
from numpy import copy

k_features = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
labels = iris_df[['species']]

# mapping species to numerical values
d = {"setosa":0, "versicolor":1, "virginica":2}
new_labels = copy(labels)
for k, v in d.items(): new_labels[labels==k] = v

training_points, test_points, training_labels, test_labels = train_test_split(k_features, new_labels, train_size=0.8, test_size=0.2)

classy = KNeighborsRegressor(n_neighbors=3)
classy.fit(training_points, training_labels)

test_predict = classy.predict(test_points)
tl = test_labels[:, -1].astype(int)
tp = test_predict[:, -1].astype(int)

k_score = f1_score(tl, tp, average='weighted')
k_score = round(k_score, 3)
print("F1 Score:", k_score)
#rate_score(k_score)

# I chose row 34 from the dataset which is a Setosa 
flower_stats = [[5.2, 4.1, 1.5, 0.1]]
species_predict = classy.predict(flower_stats)
print("Prediction. Flower Stats:", flower_stats, "Predicted Species:", species_predict)


## SUPPORT VECTOR MACHINE ##
print("\n -- SUPPORT VECTOR MACHINE -- \n")

svm_df = copy(iris_df)
svm_df = pd.DataFrame(svm_df, columns=iris_df.columns)

svm_df['species'] = np.where(svm_df['species']=='setosa', 1, 0)
svm_features = svm_df[['petal_length', 'petal_width']]
svm_labels = svm_df['species']

svm_training_points, svm_test_points, svm_training_labels, svm_test_labels = train_test_split(svm_features, svm_labels, train_size=0.8, test_size=0.2)

from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear', gamma=0.1)
svm_classifier.fit(svm_training_points, svm_training_labels)

rate_score(svm_classifier, svm_test_points, svm_test_labels)

random_points = [[[3, 2]], [[2, 0.6]]]

svm_prediction_1 = svm_classifier.predict(random_points[0])
svm_prediction_2 = svm_classifier.predict(random_points[1])
print('''\n[1] => Setosa
[0]=> Not Setosa\n''')
print("Prediction. Flower with petal length 3 and petal width 2 is predicted to be:", svm_prediction_1)
print("Prediction. Flower with petal length 2 and petal width 0.6 is predicted to be:", svm_prediction_2)

w = svm_classifier.coef_[0]
m = -w[0]/w[1]
c = -1*svm_classifier.intercept_[0]/w[1]
print("Line Equation: y =", m, "x +", c)

svm_x = np.linspace(0,7)
plt.scatter(x=svm_df['petal_length'], y=svm_df['petal_width'], c=svm_df['species'], cmap=plt.cm.coolwarm, label='Data Points')
plt.plot(svm_x, m*svm_x+c, color='Black', label='SVM Boundary')
plt.scatter(x=random_points[:][0], y=random_points[:][1], color='Green', label='Predictions', marker='s')
plt.ylim(0, 2.75)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal width and petal length of plants, grouped by species (Setosa or Other?)")
plt.legend()
plt.show()


## DECISION TREE ##
print("\n -- DECISION TREE -- \n")

tree_features = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
tree_features = np.array(tree_features)

# mapping species to numerical values
d = {"setosa":0, "versicolor":1, "virginica":2}
tree_labels = copy(labels)
for k, v in d.items(): tree_labels[labels==k] = v

tree_labels = np.array(tree_labels)
tree_labels_list = []
for i in range(len(tree_labels)):
    tree_labels_list.append(tree_labels[i][0])

tree_x_train, tree_x_test, tree_y_train, tree_y_test = train_test_split(tree_features, tree_labels_list)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(tree_x_train, tree_y_train)

rate_score(tree_classifier, tree_x_test, tree_y_test)

print('''\n[0] => Setosa
[1] => Versicolor
[2] => Virginica\n''')

# I chose row 142 from the dataset which is a Virginica Flower (2)
tree_prediction = tree_classifier.predict([[6.7, 3.1, 5.6, 2.4]])
print("Prediction. The flower with attributes: [6.7, 3.1, 5.6, 2.4] is predicted to be:", tree_prediction)