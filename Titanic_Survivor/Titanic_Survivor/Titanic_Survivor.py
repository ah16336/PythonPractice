import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#import math

## Loading Data

passengers = sns.load_dataset("titanic")

## Cleaning Data

# removing null values for age by filling it with mean age
mean_age = passengers.age.mean()
mean_age = round(mean_age)
passengers['age'].fillna(value=mean_age, inplace=True)

# making gender numerical
passengers['sex'] = np.where(passengers['sex'] == 'female', 1, 0)

# organising classes of passengers with new columns
passengers['FirstClass'] = np.where(passengers['pclass'] == 1, 1, 0)
passengers['SecondClass'] = np.where(passengers['pclass'] == 2, 1, 0)

## Making training and validation sets

# defining features that will control output
features = passengers[['sex', 'age', 'FirstClass', 'SecondClass']]
# the results
survival = passengers['survived']

x_train, x_test, y_train, y_test = train_test_split(features, survival, train_size=0.8, test_size=0.2)

## Normalising

scaler = StandardScaler()
train_features = scaler.fit_transform(x_train)
test_features = scaler.transform(x_test)

## Creating the model

model = LogisticRegression()
model.fit(train_features, y_train)

## Scoring model

print(model.score(test_features, y_test))

## Analysing coefficients

# shows us the variables that have the greatest impact on whether a passenger survives or not
print(model.coef_)

## Making sample passengers

# want to see whether these people would survive
# features in order: "Female?", "Age?", "First Class?", "Second Class?". Where all but age are yes or no.
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Beth = np.array([1.0,21.0,0.0,1.0])

sample_passengers = np.array([Jack, Rose, Beth])

## Normalising features of sample passengers

sample_passengers = scaler.transform(sample_passengers)

## Make predictions

predictions = model.predict(sample_passengers)
probs = model.predict_proba(sample_passengers)

print("Live? ", predictions)
print("How bad? ", probs)