```python
import numpy as np
```


```python
# first numpy array
first_numpy_array = np.array([1,2,3,4])
```


```python
# printing our first array
print(first_numpy_array)
```


```python
# array with only zeroes
array_with_zeros = np.zeros((3, 3))
print(array_with_zeros)
```


```python
# array with only ones
array_with_ones = np.ones((3, 3))
print(array_with_ones)
```


```python
# array with empty values
array_with_empty = np.empty((2, 3))
print(array_with_empty)
```


```python
# array with arrange method
np_arange = np.arange(12)
```


```python
# print array
print(np_arange)
```


```python
# reshape method to change/create the array
np_arange.reshape(3, 4)
```


```python
# linspace method for linearly(equal) spaced data elements
np_linspace = np.linspace(0, 6, 4)      # np.linspace(first_element, last_element, total_number_of_equidistant_elements)
print(np_linspace)
```


```python
# one-dimensional array
oneD_array = np.arange(15)
print(oneD_array)
```


```python
# two-dimensional array
twoD_array = oneD_array.reshape(3, 5)
print(twoD_array)
```


```python
# three-dimensional array
threeD_array = np.arange(27).reshape(3,3,3)
print(threeD_array)
```


```python
# ndim method of ndarray displays the no of dimensions of the array
print(oneD_array.ndim)
print(threeD_array.ndim)
print(twoD_array.ndim)
```


```python
# shape method of ndarray

print(threeD_array.shape)
print(twoD_array.shape)
print(oneD_array.shape)

# output = (no_of_rows, no_of_cols, no_of_rank)
```


```python
# size method of ndarray displays the no of elements in array

print(twoD_array.size)
print(oneD_array.size)
print(threeD_array.size)
```


```python
# dtype method describes the type of elements in array

int_array = np.arange(10)
float_array = np.array(10, dtype="float")
string_array = np.array('n', dtype="|S5")

print(int_array)
print(int_array.dtype)
print(float_array.dtype)
print(string_array.dtype)
```

# Basic oprations in NumPy


```python
first_trial_cycle = [10, 15, 17, 26]
second_trial_cycle = [12, 11, 21, 24]
```


```python
np_first_trial_cycle = np.array(first_trial_cycle)
np_second_trial_cycle = np.array(second_trial_cycle)
```


```python
first_trial_cycle + first_trial_cycle  # adding the elements after first elements
```


```python
np_first_trial_cycle + np_first_trial_cycle   # addition of array elements of 2 arrays
```


```python
np.add(45,10)
```


```python
np.subtract(45,10)
```

### An example


```python
np_daily_wage = np.array([7,9,13,8,11]) * 15
print(np_daily_wage)
```


```python
sum(np_daily_wage)
```

### Comparison Operations


```python
# create ndarray for weekly hours, 5 consecutive hours data
np_weekly_hours = np.array([23,41,55,47,38])
```


```python
# week with more than 40 hours
np_weekly_hours[np_weekly_hours>40]
```


```python
# Week with more not equals to 40 hours
np_weekly_hours[np_weekly_hours!=40]
```

### Logical operations


```python
# Logical AND operation
np.logical_and(np_weekly_hours>20, np_weekly_hours<50)
```


```python
# Logical NOT operation
np.logical_not(np_weekly_hours>35)
```


```python
# The corresponding rows:cols in ndarray is specified by below:

# (0,0) (0,1) (0,2) (0,3)
# (1,0) (1,1) (1,2) (1,3)
```


```python
# Slicing array data where 1 is inclusive but 3 is not
# Use : to select all rows
# 1 -> starting index ,3 -> ending index-1

# np_trial_data_1 = np.array([[1,2,3],[4,5,6], [7,8,9]])
# print(np_trial_data_1)

np_trial_data = np.zeros((3, 3))
trial_data = np_trial_data[:,1:3]
print(trial_data)
print("\n")
```


```python
# print all elements using for loop
for i in np_trial_data:
    print(i)
```

### Indexing with Boolean arrays
### Conditional Arguments in ndarray


```python
# Test scores of students in 4 different subjects 
test_scores = np.array([[83,71,57,63],[54,68,81,45]])
```


```python
# Passing score should be >60. So, given condition accordingly
pass_score = test_scores>60
print(pass_score)
# displays elements which fit criteria
```


```python
# prints the elements which fit the criteria
test_scores[pass_score]
```

### Universal functions


```python
# sqrt
# cos
# sin
# floor
# exp

np_arr = np.sqrt([2,4,9,16])
print(np_arr)
```


```python
from numpy import pi
np.cos(0)
```


```python
np.sin(pi/2)
```


```python
np.cos(pi)
```


```python
np.exp([0, 1, 5])
```


```python
np.floor([1.3, 5.2, 3.7, -0.4, -10.0])
```


```python
# reshaping :
# 1. split
# 2. flatten
# 3. resize
# 4. reshape
# 5. stack
```


```python
np_shape_manipulation = np.array([[23,1,54,34,76,23,67,40],[20,53,23,46,23,56,34,89]])
print(np_shape_manipulation)
```


```python
np_shape_manipulation.ravel()  # in single line
```


```python
np_shape_manipulation.reshape(4,4)  # in array with rows,cols = 4,4
```


```python
np_shape_manipulation.resize(8,2)  # 
print(np_shape_manipulation)
```


```python
np.hsplit(np_shape_manipulation, 2)   # splits the array into 2
```


```python
np_arr_1 = np.array([12,45,8,23,56,34])
np_arr_2 = np.array([2,32,57,13,76,80])
```


```python
np.hstack((np_arr_1, np_arr_2))
```


```python
np_arr_1 * np_arr_2
```


```python
var = 3.  # only if any one size is same or one of them has size is 1
np_arr_1 * var  # broadcasting multiplication in ndarray
```

### transpose method


```python
np_shape_manipulation
```


```python
np_shape_manipulation.transpose()
```


```python
np_inverse = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(np_inverse)
```


```python
np.linalg.inv(threeD_array)   # only applied to square matrix
# np.linalg.inv(np_inverse)
# np.linalg.inv(array_with_ones)
```


```python
np.trace(np_inverse)   # sum of one diagonal from left to right
```
