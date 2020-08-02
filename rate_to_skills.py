# Rate To Skills

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

# Spliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
 
## Multiple Regressor
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc.inverse_transform(regressor.predict(X_train))

## Polynomial Regressor
# Polynomial Feature
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree = 5)

# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
X_poly = polynomial_features.fit_transform(X)
reg = LinearRegression()
reg.fit(X_poly, y)

y_pred = sc.inverse_transform(reg.predict(X_poly))


# New Skills Predict
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
new_skills_pred = sc.inverse_transform(reg.predict(new_skills_poly))





############## XGBoost linear regressor ##############
import xgboost as xgb
data_dmatrix = xgb.DMatrix(data = X, label = y)

# Fitting XGBoost 
xg_reg = xgb.XGBRegressor(objective ='reg:linear', 
                          colsample_bytree = 0.3, 
                          learning_rate = 0.1, 
                          max_depth = 5, 
                          alpha = 10, 
                          n_estimators = 10)

xg_reg.fit(X_train, y_train)

y_pred = xg_reg.predict(X_train)

new_skills_Pred = xg_reg.predict(new_skills)
