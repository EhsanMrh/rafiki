# Rate To Skills Neural Network

# Importing the libraries
import numpy as np
import pandas as panda
import matplotlib.pyplot as plt

# Importing the dataset
dataset = panda.read_csv('./Datasets/skills.csv', delimiter = ',')
skills = dataset["skills"].values


# Cleaning the text
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] 

for i in range(0, len(dataset["skills"].values)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['skills'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, [2]]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
y = sc.fit_transform(y)

# Polynomial Feature
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree = 2)
X_poly = polynomial_features.fit_transform(X)

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialing the ANN
regressor = Sequential();

# Adding the input layer and the first hidden layer
input_number = len(X_poly[2])
regressor.add(Dense(output_dim = input_number, init = 'uniform', activation = 'sigmoid', input_dim = input_number))

# Adding the seconde layer
regressor.add(Dense(output_dim = 65, init = 'uniform', activation = 'sigmoid'))

# Adding the output layer
regressor.add(Dense(output_dim = 1, init = 'uniform'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the training set
regressor.fit(X_poly, y, batch_size = 10, nb_epoch = 100)



# Test Set
dataset_test = panda.read_csv('./Datasets/Skills_Test.csv', delimiter = ',')

BOW = [] 

for j in range(0, len(dataset_test["skills"].values)):
    review = re.sub('[^a-zA-Z]', ' ', dataset_test['skills'][j])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    BOW.append(review)

new_skills = BOW
new_skills = cv.transform(new_skills).toarray()

new_skills_poly = polynomial_features.transform(new_skills)
new_skills_pred = sc.inverse_transform(regressor.predict(new_skills_poly))



















