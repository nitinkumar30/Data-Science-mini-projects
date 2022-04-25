```python
#import necessary libraries
import numpy as np
import pandas as pd
```


```python
# import scikit learn dataset already loaded in scikit-learn
from sklearn.datasets import load_boston
boston_dataset = load_boston()
```


```python
# use built in methods to explore and understand the data
print(boston_dataset['DESCR'])
```


```python
# print the features of the dataset
print(boston_dataset['feature_names'])
```


```python
# store data into dataframe
df_boston = pd.DataFrame(boston_dataset.data)
```


```python
# set features as columns on the dataframe
df_boston.columns = boston_dataset.feature_names
```


```python
# view first 5 observation
df_boston.head()
```


```python
# print dataset matrix (observation and features matrix)
df_boston.shape
```


```python
# print dataset target or response shape
boston_dataset.target.shape
```


```python
# view target or response
boston_dataset['target']
```

## Machine Learning - Linear Regression


```python
# import required libraries
import numpy as np
import pandas as pd
```


```python
# import boston dataset
from sklearn.datasets import load_boston
boston_dataset = load_boston()
```


```python
# create pandas dataset and store the data
df_boston = pd.DataFrame(boston_dataset.data)
df_boston.columns = boston_dataset.feature_names
```


```python
df_boston.head()
```


```python
# append price, target, as a new column to the dataset
df_boston['Price'] = boston_dataset.target
```


```python
# print top 5 observations
df_boston.head()
```


```python
# assign features on X-axis
X_features = boston_dataset.data
```


```python
# assign target on Y-axis
Y_target = boston_dataset.target
```


```python
# import linear model - the estimator
from sklearn.linear_model import LinearRegression
lineReg = LinearRegression()
```


```python
# fit data into the estimator
lineReg.fit(X_features,Y_target)
```


```python
# print the intercept
print('The estimated intercept %.2f '%lineReg.intercept_)
```


```python
# print the coefficient
print('The coefficient is %d' %len(lineReg.coef_))
```


```python
# train model split the whole dataset into train and test datasets
# from sklearn import cross_validation
# https://stackoverflow.com/questions/53978901/importerror-cannot-import-name-cross-validation-from-sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_features,Y_target)
```


```python
# print the dataset shape
print(boston_dataset.data.shape)
```


```python
# print shapes of training and testing data sets
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
```


```python
# fit the training sets into the model
lineReg.fit(X_train, Y_train)
```


```python
# The mean square error or residual sum of squares
print('MSE value is %.2f ' %np.mean(lineReg.predict(X_test)-Y_test) **2)
```


```python
# calculate variance
print('Variance score is %.2f' %lineReg.score(X_test, Y_test))
```

## Supervised learning models : Logistic Regression


```python
# import necessary modules
import numpy as np
import pandas as pd
```


```python
# import sklearn load dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()
```


```python
# display the dataset
type(iris_dataset)
```


```python
# view information using dataset built in method DESCR(describe)
print(iris_dataset.DESCR)
```


```python
# View features
print(iris_dataset.feature_names)
```


```python
# View target
print(iris_dataset.target)
```


```python
# Find number of observations
print(iris_dataset.data.shape)
```


```python
# Assign features data to x-axis
X_feature = iris_dataset.data
```


```python
# Assign target data to y-axis
Y_target = iris_dataset.target
```


```python
# View the shape of both axis
print(X_feature.shape)
print(Y_target.shape)
```

### KNN model importing from sklearn


```python
# First use KNN classifier method - import it from sklearn
from sklearn.neighbors import KNeighborsClassifier
```


```python
# instantiate the knn estimator - object used to instantiate the class of a learning model is called an estimator
knn = KNeighborsClassifier(n_neighbors=1)
```


```python
# print the knn
print(knn)
```


```python
# fit data into knn model(estimator)
knn.fit(X_feature, Y_target)
```


```python
# create object with new values for prediction
X_new = [[3,5,4,1],[5,3,4,2]]
```


```python
# PRedict the outcome for the new observation using knn classifier
knn.predict(X_new)
```


```python
# Use logistic regression estimator
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
```


```python
# fit data into the Logistic regression estimator
logReg.fit(X_feature, Y_target)
```


```python
# predict the outcome using Logistic Regression estimator 
logReg.predict(X_new)
```

## Unsupervised Learning Models : Clustering

### KMeans Clustering 


```python
# import required libraries
import numpy as np

# import KMeans class from sklearn.cluster
from sklearn.cluster import KMeans

# import make_blobs dataset from sklearn.cluster
from sklearn.datasets import make_blobs
```


```python
# Define number of smaples
n_samples = 300

# Define random state value to initialize the center
random_state = 20

# define number of feature as 5
X,y = make_blobs(n_samples=n_samples, n_features=5, random_state=None)

# define number of cluster to be formed as 3 and in random state and fit features into the model 
predict_y = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

# print the estimator prediction
predict_y
```

### Unsupervised Learning Models : Dimensionality Reduction 
> Helps cut down dimentions without losing any data from dataset

**Techniques used for dimensionality reduction :**
- Drop data columns with missing values
- Drop data columns with low variance
- Drop data columns with high corelations
- Apply statistical functions - PCA(Principal Component Analysis)

## PCA implementation


```python
# import required library PCA
from sklearn.decomposition import PCA

# import the dataset
from sklearn.datasets import make_blobs
```


```python
# Define sample and random state
n_sample = 20
random_state = 20
```


```python
# Generate the dataset with 10 features (dimension)
X,y = make_blobs(n_samples=n_sample, n_features=10, random_state=None)
```


```python
# View the shape of the dataset
X.shape
```


```python
# Define the PCA estimator with number of reduced components 
pca = PCA(n_components=3)
```


```python
# Fit the data into the PCA estimator
pca.fit(X)
print(pca.explained_variance_ratio_)
```


```python
# Print the first PCA component 
first_pca = pca.components_[0]
print(first_pca)
```


```python
# Transform the fitted data using transform method
pca_reduced= pca.transform(X)
```


```python
# View the reduced shape (lower dimension)
pca_reduced.shape
```

## Pipeline - Build pipeline using scikit-learn

#### Import the required libraries and models(estimators)


```python
# import pipeline class
from sklearn.pipeline import Pipeline

# import linear estimator 
from sklearn.linear_model import LinearRegression

# import pca estimator for dimensionality reduction
from sklearn.decomposition import PCA
```

#### Chain the estimators together


```python
estimator = [('dim_reduction',PCA()), ('linear_model',LinearRegression())]
```

#### Put the chain of estimators in a pipeline object


```python
pipeline_estimator = Pipeline(estimator)
```

#### Check the chain of estimators


```python
pipeline_estimator
```

#### View the first step


```python
pipeline_estimator.steps[0]
```

#### View second step


```python
pipeline_estimator.steps[1]
```

#### View all the steps in pipeline


```python
pipeline_estimator.steps
```

### Model Persistence and Evaluation


```python
# Import required libraries and dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()
```


```python
# View feature names of the dataset
iris_dataset.feature_names
```


```python
# View target of the dataset
iris_dataset.target
```


```python
# Define features and target objects
X_feature = iris_dataset.data
Y_target = iris_dataset.target
```


```python
# Create object with new values for prediction
X_new = [[3,5,4,1],[5,3,4,2]]
```


```python
# Use logistical regression estimator
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
```


```python
# Fit data into the logistic regression estimator
logReg.fit(X_feature, Y_target)
```


```python
# Predict the outcome using logistic regression estimator
logReg.predict(X_new)
```


```python
# import library for model persistance
import pickle as pkl
```


```python
# Use dumps method to persist the model
persist_model = pkl.dumps(logReg)
persist_model
```


```python
# Use joblib.dump to persist the model to a file
# from sklearn.externals import joblib
import joblib
joblib.dump(logReg, 'regresfilename.pkl')
```


```python
# Create new estimator from the saved model
new_logreg_estimator = joblib.load('regresfilename.pkl')
```


```python
# View the new estimator
new_logreg_estimator
```


```python
# Validate and use new estimator to predict
new_logreg_estimator.predict(X_new)
```

### Metric functions to evaluate accuracy of your model's predictions

1. Classification : metrics.accuracy_score | metrics.average_precision_score
2. Clustering : metrics.adjusted_rand_score
3. Regression : metrics.mean_absolute_error | metrics.mean_squared_error | metrics.median_absolute_error


```python

```
