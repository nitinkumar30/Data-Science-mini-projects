# SciPy in Data Science

### single integration


```python
from scipy.integrate import quad   # import scipy library
```


```python
def integrateFunction(x):
    return x   # function returns the argument value
```


```python
quad(integrateFunction,0,1)  # quad with limits set to 0 to 1
```


```python
def integrateFunc(x,a,b):  # another functions with multiple arguments
    return x*a+b
```


```python
a=3
b=2
```


```python
quad(integrateFunc,0,1,args=(a,b))   # providing the desired arguments
```

### double integration


```python
import scipy.integrate as integrate
```


```python
def f(x,y):
    return x+y
integrate.dblquad(f,0,1,lambda x:0, lambda x:2)   # double integration
```

### optimization 
> Used to improve the performance of the system mathematically by fine tuning the process parameters


```python
import numpy as np
from scipy import optimize
```


```python
def f(x):
    return x**2 + 5*np.sin(x)   # define function for the x square plus 5 sin(x)
```


```python
minimal_value = optimize.minimize(f,x0=0,method='bfgs',options={'disp':True})   # perform optimize minimize function
                                                                                # using bfgs method and options
```


```python
minimalValueWithoutOutput=optimize.minimize(f,x0=0,method='bfgs')  # perform optimize minimize function
minimalValueWithoutOutput                                          # using bfgs method and without options
```


```python
from scipy.optimize import root
```


```python
def rootFunc(x):
    return x + 3.5* np.cos((x))   # define function
```


```python
rootValue = root(rootFunc,0.3)   # pass x value in argument for root
rootValue       # function values and array values 
```

### Inverse of matrix


```python
import numpy as np
from scipy import linalg

matrix = np.array([[10,3],[2,8]])
matrix
```


```python
type(matrix)
```


```python
linalg.inv(matrix)   # inversing the elements of matrix
```

### finding determinant


```python
linalg.det(matrix)   # method to find determinant
```

### Solve linear functions


```python
# Linear Equations:
# 2x + 3y + z = 21
# -x + 5y + 4z = 9
# 3x + 2y + 9z = 6
```


```python
numArray = np.array([[2,3,1],[-1,5,4],[3,2,9]])
numArrayValue = np.array([21,9,6])
```


```python
linalg.solve(numArray, numArrayValue)    # method to solve linear equations
```

### Single value decomposition


```python
matrix.shape
```


```python
linalg.svd(matrix)
```


```python
# (array([[-0.84330347, -0.53743768],
#         [-0.53743768,  0.84330347]]),    <- Unitary matrix

#  array([11.70646059,  6.32129579]),     <- Sigma or Square root of eigenvalues

#  array([[-0.8121934 , -0.58338827],
#         [-0.58338827,  0.8121934 ]]))   <- VH is values collected into unitary matrix
```

## Eigenvalues & EigenVector


```python
# import the required libraries
import numpy as np
from scipy import linalg
```


```python
# test_data matrix
test_rating_data = np.array([[5,8],[7,9]])
eigenValues, eigenVector = linalg.eig(test_rating_data)
first_eigen, second_eigen = eigenValues
```


```python
# print eigenvalues
first_eigen, second_eigen
```


```python
# print first eigenvector
eigenVector[:,0]
```


```python
# print second eigenvector
eigenVector[:,1]
```

## Scipy sub-package - statistics


```python
from scipy.stats import norm   # import norm for normal distribution 
```


```python
norm.rvs(loc=0,scale=1,size=10)   # rvs for ransom variables
```


```python
norm.cdf(5,loc=1,scale=2)   # cds for Commulative Distribution Function
```


```python
norm.pdf(9,loc=0,scale=1)   # pdf for Probability Density Function for random distribution
```
