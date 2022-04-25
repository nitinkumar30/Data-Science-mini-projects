## Steps to create plot

- Import the required libraries
- Define or import the required dataset
- Set the plot parameters
- Display the plot


```python
# import the libraries
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
```


```python
# generate random numbers
randomNumbers = np.random.rand(10)
```


```python
# view them
randomNumbers
```


```python
# select the style of the plot
style.use('ggplot')

# plot the random number
plt.plot(randomNumbers, 'g', label='Line one', linewidth=3)  
# plt.plot(dataName, 'colour_first_letter', label='line_definition', linewidth=widht_of_line_drawn)

# x-axis is number of random numbers
plt.xlabel('Range')

# y-axis is actual random numbers
plt.ylabel('Numbers')

# title of the plot
plt.title('First plot')

plt.legend()
plt.show()
```

## Another example


```python
# import the required libraries
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
```


```python
# website traffic data
# number of users on the website
web_customers = [123, 645,950, 1290,1630,1450,1034,1295,465,205,80]

# time distribution
time_hrs = [7,8,9,10,11,12,13,14,15,16,17]
```


```python
# select style of graph
style.use('ggplot')

# plot the traffic website data (X-axis hrs and Y-axis as numer of users)
plt.plot(time_hrs, web_customers)

# set the title of plot
plt.title('Website traffic')

# set label for x-axis
plt.xlabel('Time(in hrs)')

# set label for y-axis
plt.ylabel('Visitors')

# display plot
plt.show()
```


```python
# select style of graph
style.use('ggplot')

# plot the traffic website data (X-axis hrs and Y-axis as numer of users)
plt.plot(time_hrs, web_customers, color='c', linestyle='--', linewidth=3, label='web traffic')

# set the title of plot
plt.title('Website traffic')

# set label for x-axis
plt.xlabel('Time(in hrs)')

# set label for y-axis
plt.ylabel('Visitors')

# display plot
plt.legend()
plt.show()
```


```python
# select style of graph
style.use('ggplot')

# plot the traffic website data (X-axis hrs and Y-axis as numer of users)
plt.plot(time_hrs, web_customers, label='web traffic', linewidth=2)
plt.axis([6.5,17.5,50,2000])  # set axis

# set the title of plot
plt.title('Website traffic')

# set label for x-axis
plt.xlabel('Time(in hrs)')

# set label for y-axis
plt.ylabel('Visitors')

# display plot
plt.legend()
plt.show()
```


```python
# select style of graph
style.use('ggplot')

# plot the traffic website data (X-axis hrs and Y-axis as numer of users)
plt.plot(time_hrs, web_customers)

# set the title of plot
plt.title('Website traffic')

# Annotate
plt.annotate('Max', ha='center',va='bottom', xytext=(8,1500),xy=(11,1630),arrowprops={'facecolor':'blue'})
# plt.annonate('annotation_text','text_position','arrow_position')   # ha - horizontal allignment, va - vertical allignment, 
# set label for x-axis
plt.xlabel('Time(in hrs)')

# set label for y-axis
plt.ylabel('Visitors')

# display plot
plt.show()
```

### Multiple plots


```python
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
```


```python
# 03:03/10:01
# website traffic data
# number of visitors on the wbsite
# monday web traffic
web_monday = [123,645,950,1290,1630,1450,1034,1295,465,205,80]
# tuesday web traffic
web_tuesday = [95,680,889,1145,1670,1323,1119,1265,510,310,110]
# wednesday web traffic
web_wednesday = [105,630,700,1006,1520,1124,1239,1380,580,610,230]
# time distribution hourly 
time_hrs = [7,8,9,10,11,12,13,14,15,16,17]
```


```python
# select the style of the plot
# --------------------------------------  
# plot the website traffic data (x-axis as hrs & y-axis as number of users)
# plot monday traffic with red colour
plt.plot(time_hrs, web_monday, 'r', label='Monday', linewidth=1)

# plot tueday traffic with green colour
plt.plot(time_hrs, web_tuesday, 'g', label='Tuesday', linewidth=1)

# plot wednesday traffic with blue colour
plt.plot(time_hrs, web_wednesday, 'b', label='Wednesday', linewidth=1)

plt.axis([6.5, 17.5, 50, 2000])

# set title of plot
plt.title('Web traffic plotting')

# set label for x-axis
plt.xlabel('Hrs')

# set label for y axis
plt.ylabel('Number of users')

plt.legend()
plt.show()
```

### Sub plot

> subplot(m, n, p)    # m-by-n grid and creates an axis for the subplot in the position specified by p


```python
# import required libraries
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
```


```python
# define temp, wind, humidity, precipitation data and time hours data
temp_data = [91, 74, 91, 98, 77, 85, 97, 76, 98, 83, 93, 79, 96, 85, 97, 75, 85, 97, 100, 98, 99, 89, 70, 100]
wind_data = [17, 8, 13, 24, 16, 13, 11, 13, 14, 9, 24, 11, 11, 10, 19, 8, 9, 10, 21, 25, 15, 15, 9, 24]
time_hrs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
humidity_data = [64, 53, 72, 83, 81, 51, 76, 83, 55, 67, 85, 85, 64, 74, 70, 60, 51, 78, 69, 78, 58, 75, 85, 57]
precipitation_data = [50, 34, 40, 60, 13, 39, 59, 68, 66, 51, 46, 24, 47, 57, 11, 64, 41, 70, 23, 57, 66, 62, 69, 61]

# Code used for data:-

# import random
# rslt = [random.randint(8,25) for i in range(24)]
# # rslt  = [i for i in range(1,25)]
# print(rslt)
```


```python
# draw subplots for (1,2,1) and (1,2,2)
plt.figure(figsize=(8,4))
plt.subplots_adjust(hspace=.25)
plt.subplot(1,2,1)
plt.title('Temp')
plt.plot(time_hrs, temp_data, color='b', linestyle='-', linewidth=1)
plt.subplot(1,2,2)
plt.title('Wind')
plt.plot(time_hrs, wind_data, color='r', linestyle='-', linewidth=1)
```


```python
# draw subplots for (2,1,1) and (2,1,2)
plt.figure(figsize=(6,6))
plt.subplots_adjust(hspace=.25)
plt.subplot(2,1,1)
plt.title('Humidity')
plt.plot(time_hrs, humidity_data, color='b', linestyle='-', linewidth=1)
plt.title('Precipitation')
plt.plot(time_hrs, precipitation_data, color='r', linestyle='-', linewidth=1)
plt.show()
```


```python
# draw subplots for (2,1,1) and (2,1,2)
plt.figure(figsize=(6,6))
plt.subplots_adjust(hspace=.25)
plt.subplot(2,1,1)
plt.title('Humidity')
plt.plot(time_hrs, humidity_data, color='b', linestyle='-', linewidth=1)
plt.subplot(2,1,2)
plt.title('Precipitation')
plt.plot(time_hrs, precipitation_data, color='r', linestyle='-', linewidth=1)
plt.show()
```


```python
# draw subplots for (2,2,1), (2,2,2), (2,2,3) and (2,2,4)
plt.figure(figsize=(9,9))
plt.subplots_adjust(hspace=.3)
plt.subplot(2,2,1)
plt.title('Temp (F)')
plt.plot(time_hrs, temp_data, color='g', linestyle='-', linewidth=1)

plt.subplot(2,2,2)
plt.title('Wind (mph)')
plt.plot(time_hrs, wind_data, color='r', linestyle='-', linewidth=1)

plt.subplot(2,2,3)
plt.title('Humidity (%)')
plt.plot(time_hrs, humidity_data, color='b', linestyle='-', linewidth=1)

plt.subplot(2,2,4)
plt.title('Precipitation (%)')
plt.plot(time_hrs, precipitation_data, color='c', linestyle='-', linewidth=1)
```

### Histogram & Scatter plots


```python
# import the boston dataset from sklearn library
from sklearn.datasets import load_boston

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
```


```python
# load boston dataset
boston_data = load_boston()
```


```python
# view boston dataset
print(boston_data.DESCR)
```


```python
# define x-axis for the data
x_axis = boston_data.data
```


```python
# define y-axis for the target
y_axis = boston_data.target
```


```python
# plot histogram
style.use('ggplot')
plt.figure(figsize=(7,7))
plt.hist(y_axis, bins=50)
plt.xlabel('Price in 1000s USD')
plt.ylabel('Number of houses')
plt.show()
```


```python
# plot scatter plot
style.use('ggplot')
plt.figure(figsize=(7,7))
plt.scatter(boston_data.data[:,5],boston_data.target, color='b')
plt.xlabel('Price in 1000s USD')
plt.ylabel('Number of houses')
plt.show()
```

### Heatmap using matplotlib library


```python
# import matplotlib library
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline

# import seaborn library
import seaborn as sns
```


```python
# load flights data from sns dataset (built-in)
flight_data = sns.load_dataset('flights')
```


```python
# view top 5 data
flight_data.head()
```


```python
# use pivot method to rearrange the dataset
flight_data = flight_data.pivot('month', 'year', 'passengers')
```


```python
# view the dataset
flight_data
```


```python
# use heatmap method to generate the heatmap of the flights data
sns.heatmap(flight_data)
```

### Piechart


```python
# import required libraries
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
```


```python
# job data in precentile
job_data = ['40','20','17','8','5','10']

# define label as different departments
labels = 'IT','Finance','Marketing','Admin','HR','Operations'

# explode the 1st slide which is IT
explode = (0.05,0,0,0,0,0)

# draw the pie chart and set parameters
plt.pie(job_data, labels=labels, explode=explode)

# show the plot
plt.show()
```
