import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

flags = pd.read_csv("flags.csv", header=0)

# exploring dataset

#print(flags.columns)
#print(flags.head())

# values we will want to predict later (where the country is located on the planet)
labels = flags[['Landmass']]
# the features that we will use to make the prediction (a colour means whether that colour featured in the flag or not)
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

# creating training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

# testing which depth we want for our tree
best_score = 0
best_depth = 0
scores = []
for i in range(1, 21):
  classifier = DecisionTreeClassifier(random_state=1, max_depth=i)
  classifier.fit(train_data, train_labels)
  score = classifier.score(test_data, test_labels)
  scores.append(score)
  if best_score < score:
    best_score = score
    best_depth = i

# printing the various scores to visualise which tree depth to use
plt.plot(range(1, 21), scores)
plt.show()
print(best_depth, ",", best_score)
