import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def rate_score(score):
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

plt.scatter(x, y)
plt.show()

x = x.values.reshape(-1, 1)
line_x_train, line_x_test, line_y_train, line_y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

from sklearn.linear_model import LinearRegression

line = LinearRegression()
line.fit(line_x_train, line_y_train)

print("Line Equation: y =", line.coef_, "x +", line.intercept_)

score = line.score(line_x_test, line_y_test)
print("Score:", score)
rate_score(score)

unknowns = np.array([8.2, 1.9, 5.7, 2.3, 4.5])
unknowns = unknowns.reshape(-1, 1)

predictions = line.predict(unknowns)

plt.scatter(unknowns, predictions)
plt.show()


## MULTIPLE LINEAR REGRESSION ##
print("\n -- MULTIPLE LINEAR REGRESSION -- \n")

features = iris_df[['sepal_length', 'sepal_width', 'petal_width']]
# y is unchanged

mlr_x_train, mlr_x_test, mlr_y_train, mlr_y_test = train_test_split(features, y, train_size=0.8, test_size=0.2)
mlr = LinearRegression()
mlr.fit(mlr_x_train, mlr_y_train)

# Which variable had the biggest impact?
print("Coefficients:", mlr.coef_)

score_mlr = mlr.score(mlr_x_test, mlr_y_test)
print("Score:", score_mlr)
rate_score(score_mlr)

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
print("F1 Score:", k_score)
rate_score(k_score)


flower_stats = [[5.2, 4.1, 1.5, 0.1]]
species_predict = classy.predict(flower_stats)
print("Prediction. Flower Stats:", flower_stats, "Predicted Species:", species_predict)