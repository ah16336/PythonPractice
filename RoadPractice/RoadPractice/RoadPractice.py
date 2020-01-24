import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("road_data.csv")

#print(df.head())

features = df[['Vehicle_Reference', 'Casualty_Class', 'Sex_of_Casualty', 'Age_of_Casualty']]
features = np.array(features)
labels = df[['Casualty_Severity']]
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(features, labels)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

#tree_score = tree.score(x_test, y_test)
#print(tree_score)

crash = [[1, 2, 2, 21]]

prediction = tree.predict(crash)
print(prediction)

mean_age = df.Age_of_Casualty.mean()
mean_gender = df.Sex_of_Casualty.mean()

print("Expected Age of Casualty:", mean_age)
print("Expected Gender (1 is Male and 2 is Female):", mean_gender)

stats = pd.read_csv("STATS19.csv")

lr_features = stats[["Day_of_Week", "Weather_Conditions", "Light_Conditions", "1st_Road_Class", "Speed_limit", "Urban_or_Rural_Area"]]

stats['Accident_Severity'] = np.where(stats["Accident_Severity"] == 1, 1, 0) # fatal is 1, 0 otherwise

lr_label = stats[['Accident_Severity']]

lr_x_train, lr_x_test, lr_y_train, lr_y_test = train_test_split(lr_features, lr_label, train_size=0.8, test_size=0.2)

lr_model = LogisticRegression()
lr_model.fit(lr_x_train, lr_y_train)

print("Features:" "Day_of_Week", "Weather_Conditions", "Light_Conditions", "1st_Road_Class", "Speed_limit", "Urban_or_Rural_Area")
print(lr_model.coef_)

