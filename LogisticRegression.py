import numpy as np
from sklearn.linear_model import LogisticRegression
from exam import hours_studied_scaled, passed_exam, exam_features_scaled_train,\
    exam_features_scaled_test, passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled

# Create and fit logistic regression model here
model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)

# Save the model coefficients and intercept here
calculated_coefficients = model.coef_
intercept = model.intercept_
print(calculated_coefficients)
print(intercept)

# Predict the probabilities of passing for next semester's students here
passed_predictions = model.predict_proba(guessed_hours_scaled)


# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled_train, passed_exam_2_train)

# Predict whether the students will pass here
passed_predictions_2 = model_2.predict(exam_features_scaled_test)
print(passed_predictions_2)
print(passed_exam_2_test)

# Assign and update coefficients
coefficients = model_2.coef_
coefficients = coefficients.tolist()[0]

# Plot bar graph
plt.bar([1, 2], coefficients)
plt.xticks([1, 2], ['hours studied', 'math courses taken'])
plt.xlabel('feature')
plt.ylabel('coefficient')

plt.show()

################# project: Titantic survive

import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')

# look at data frame names
print(passengers.columns)

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female': 1, 'male': 0})

# Fill the nan values in the age column
passengers['Age'].fillna(value = round(passengers['Age'].mean()), inplace = True)

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 2 else 0)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
train_features, valid_features, train_labels, valid_labels = train_test_split(features, survival, test_size = 0.8)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scalar = StandardScaler()
train_features = scalar.fit_transform(train_features)
valid_features = scalar.transform(valid_features)

# Create and train the model
classifier = LogisticRegression()
classifier.fit(train_features, train_labels)

# Score the model on the train data
score = classifier.score(train_features, train_labels)
print(score)

# Score the model on the test data
score = classifier.score(valid_features, valid_labels)
print(score)

# Analyze the coefficients
coeff = classifier.coef_
print(coeff)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0,25,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample_passengers = scalar.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
survive_ans = classifier.predict(sample_passengers)
print(survive_ans)

survive_prob = classifier.predict_proba(sample_passengers)
print(survive_prob)