### Documentation of Pandas is [here](https://pandas.pydata.org/docs/user_guide/10min.html)


```python
# importing libraries 
import numpy as np
import pandas as pd
```

## Series is a datatype(data structure) used in Pandas for data manipulation

### passing object as argument 


```python
first_series = pd.Series(list('abcdef'))  # passing list as argument
```


```python
first_series  # index assigned automatically
```

### passing List as argument 


```python
np_names = np.array(['nitin','somi','yash','anji','rhea','ujjawal'])   # ndarray for names
```


```python
np_names_series = pd.Series(np_names)
```


```python
np_names_series
```

### passing dictionary as argument 


```python
dict_country_gdp = pd.Series([2255.225482,629.9553062,11601.63022,25306.82494,27266.40335,19466.99052,588.3691778,2890.345675,24733.62696,1445.760002,4803.398244,2618.876037,590.4521124,665.7982328,7122.938458,2639.54156,3362.4656,15378.16704,30860.12808,2579.115607,6525.541272,229.6769525,2242.689259,27570.4852,23016.84778,1334.646773,402.6953275,6047.200797,394.1156638,385.5793827,1414.072488,5745.981529,837.7464011,1206.991065,27715.52837,18937.24998,39578.07441,478.2194906,16684.21278,279.2204061,5345.213415,6288.25324,1908.304416,274.8728621,14646.42094,40034.85063,672.1547506,3359.517402,36152.66676,3054.727742,33529.83052,3825.093781,15428.32098,33630.24604,39170.41371,2699.123242,21058.43643,28272.40661,37691.02733,9581.05659,5671.912202,757.4009286,347.7456605],index=['Algeria','Angola','Argentina','Australia','Austria','Bahamas','Bangladesh','Belarus','Belgium','Bhutan','Brazil','Bulgaria','Cambodia','Cameroon','Chile','China','Colombia','Cyprus','Denmark','El Salvador','Estonia','Ethiopia','Fiji','Finland','France','Georgia','Ghana','Grenada','Guinea','Haiti','Honduras','Hungary','India','Indonesia','Ireland','Italy','Japan','Kenya', 'South Korea','Liberia','Malaysia','Mexico', 'Morocco','Nepal','New Zealand','Norway','Pakistan', 'Peru','Qatar','Russia','Singapore','South Africa','Spain','Sweden','Switzerland','Thailand', 'United Arab Emirates','United Kingdom','United States','Uruguay','Venezuela','Vietnam','Zimbabwe'])
```


```python
dict_country_gdp
```

### passing scalar as argument 


```python
scalar_series = pd.Series(5.,index=['a','b','c','d','e'])
```


```python
scalar_series
```

### Accessing elements in Series


```python
dict_country_gdp[0]    # using index no
```


```python
dict_country_gdp[0:5]   # using range of index numbers
```


```python
dict_country_gdp[4:9]   # using range of index nos from any index
```


```python
dict_country_gdp.loc['India']   # using name of country
```


```python
dict_country_gdp.iloc[0]    # using position
```

### vectorized operations in Series


```python
first_vector_series = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
second_vector_series = pd.Series([10,20,30,40,50],index=['a','b','c','d','e'])
```


```python
first_vector_series
```


```python
second_vector_series
```


```python
first_vector_series + second_vector_series
```


```python
first_vector_series = pd.Series([1,2,3,4,5],index=['e','c','b','d','a'])
```


```python
first_vector_series + second_vector_series
```


```python
first_vector_series = pd.Series([1,2,3,4,5],index=['e','c','b','f','g'])
```


```python
first_vector_series + second_vector_series
```

## Dataframes - another data structure

### Create dataframe from lists


```python
import pandas as pd
```


```python
# last 5 olympic date,place,year and no of countries participated

olympic_data_list = {'Hostcity':['London','Beijing','Athens','Sydney','Atlanta'], 'Year':[2021,2008,2004,2000,1996],
                    'No of participating countries':[205,204,201,200,197]}
```


```python
df_olympic_data = pd.DataFrame(olympic_data_list)
```


```python
df_olympic_data
```

### Create dataframe from dictionary


```python
olympic_data_dict = {'London':{2021:205},'Beijing':{2008:204}}
```


```python
df_olympic_data_dict = pd.DataFrame(olympic_data_dict)
```


```python
df_olympic_data_dict
```

### View dataframe


```python
df_olympic_data.Hostcity
```


```python
df_olympic_data.describe
```

### Create dataframe from dict of series


```python
olympic_series_participation = pd.Series([205,204,201,200,197],index=[2012,2008,2004,2000,1996])
olympic_series_country = pd.Series(['London','Beijing','Athens','Sydney','Atlanta'],
                                  index=[2012,2008,2004,2000,1996])
```


```python
df_olympic_series = pd.DataFrame({'No of participating Countries':olympic_series_participation,'Host Countries':olympic_series_country})
```


```python
df_olympic_series
```

### Create dataframe from ndarray


```python
import numpy as np
```


```python
np_array = np.array([2012,2008,2004,2000,1996])
dict_ndarray = {'Year': np_array}
```


```python
df_ndarray = pd.DataFrame(dict_ndarray)
```


```python
df_ndarray
```

### Create dataframe from dataframe object


```python
df_from_df = pd.DataFrame(df_olympic_series)
```


```python
df_from_df
```

## View and Select data in Pandas


```python
# import libraries
import numpy as np
import pandas as pd
```


```python
# create dataframe from dict of series for summer olympics : 1996 - 2012
olympic_series_participation = pd.Series([205,204,201,200,197],index=[2012,2008,2004,2000,1996])
olympic_series_country = pd.Series(['London','Beijing','Athens','Sydney','Atlanta'],
                                  index=[2012,2008,2004,2000,1996])

df_olympic_series = pd.DataFrame({'No of participating Countries':olympic_series_participation,'Host Cities':olympic_series_country})
```


```python
# Display content of the dataset
df_olympic_series
```


```python
# view dataframe describe
df_olympic_series.describe
```


```python
# view top 2 records
df_olympic_series.head(2)
```


```python
# view last 3 records
df_olympic_series.tail(3)
```


```python
# view indexes of datasets
df_olympic_series.index
```


```python
# view columns of dataset
df_olympic_series.columns
```

### Select data


```python
# select data for host city
df_olympic_series['Host Cities']
```


```python
# Another data selection no. of participating countries
df_olympic_series['No of participating Countries']
```


```python
# Select label-location based access by label
df_olympic_series.loc[2012]
```


```python
# integer-location based indexing by position
df_olympic_series.iloc[0:2]
```


```python
# integer-location based data selection by index value
df_olympic_series.iat[3, 1]
```


```python
# select data element by condition where number of participated countries are more than 200 
# HINT - Use boolean expression
df_olympic_series[df_olympic_series['No of participating Countries'] == 200]
```

## Handling missing values


```python
first_series = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
```


```python
second_series = pd.Series([10,20,30,40,50],index=['c','e','f','g','h'])
```


```python
sum_of_series = first_series + second_series
```


```python
sum_of_series
```


```python
# drop NaN (Not a Number) value from dataset
dropna_s = sum_of_series.dropna()
```


```python
dropna_s
```


```python
# Fill NaN (Not a Number) values from dataset with zeroes
fillna_s = sum_of_series.fillna(0)
```


```python
fillna_s
```


```python
# fill NaN with zeroes before performing addition operation for missing indices
fill_NaN_with_zeroes_before_sum = first_series.add(second_series, fill_value=0)
```


```python
fill_NaN_with_zeroes_before_sum
```

## Data Operations


```python
import pandas as pd
```


```python
# declare movie rating dataframe: ratings from 1 to 5 (star * rating)
df_movie_rating = pd.DataFrame({'movie 1':[5,4,3,3,2,1],'movie 2':[4,5,2,3,4,2]},index=['Tom','Jeffer','Peter','Ram','Ted','Paul'])
```


```python
df_movie_rating
```


```python
def movie_grade(rating):
    if rating==5:
        return 'A'
    if rating==4:
        return 'B'
    if rating==3:
        return 'C'
    if rating==2:
        return 'D'
    if rating==1:
        return 'E'
    else:
        return 'F'
```


```python
print(movie_grade(5))
```


```python
df_movie_rating.applymap(movie_grade)
```

## Data operations with statistical functions


```python
import pandas as pd
```


```python
df_test_scores = pd.DataFrame({'Test1':[95,84,73,88,82,61], 'Test2':[74,85,82,73,77,79]}, index=['Jack','Lewis','Patrick','Rich','Kelly','Paula'])
```


```python
df_test_scores.max()   # max function to find maximum score
```


```python
df_test_scores.mean()   # mean function to find mean of both of them
```


```python
df_test_scores.std()    # std function to find standard deviation
```

### groupby function to operate data 


```python
df_president_names = pd.DataFrame({'first':['George','Bill','Ronald','Jimmy','George'],'last':['Bush','Clinton','Regan','Carter','Washington']})

# create datafrae with president names 
```


```python
df_president_names
```


```python
first_only = df_president_names.groupby('first')
```


```python
first_only.get_group('George')
```


```python
df_president_names.sort_values('first')
```

### How to stadardise our dataset


```python
def standardize_tests(test):   # create funtion to standardize value
    return (test-test.mean())/test.std()
```


```python
standardize_tests(df_test_scores['Test1'])
```


```python
def standardize_test_scores(dataframe):    # apply function to entire dataset
    return dataframe.apply(standardize_tests)
```


```python
standardize_test_scores(df_test_scores)
```

## Pandas Data Operation - Merge, Duplicate & Concatenation


```python
# import required libraries
import pandas as pd
```


```python
# define student data with math data
df_student_math = pd.DataFrame({'student':['Tom','Jack','Dan','Ram','Jeff','David'],'ID':[10,56,31,85,9,22]})
```


```python
# define student data from science data
df_student_science = pd.DataFrame({'student':['Tom','Ram','David'],'ID':[10,12,22]})
```


```python
# merge both data to form single dataframe with math & science data
pd.merge(df_student_math,df_student_science)
```


```python
# merge with key on student
pd.merge(df_student_math,df_student_science, on='student')
```


```python
# merge left join on key ID & also fill NaN values with x
pd.merge(df_student_math,df_student_science, on='ID',how='left').fillna('X')
```


```python
# concat data of both subjects
pd.concat([df_student_math, df_student_science],ignore_index=True)
```


```python
# define new data frame with student survey data
df_student_survey_data = pd.DataFrame({'student':['Tom','Jack','Tom','Ram','Jeff','Jack'],'ID':[10,56,10,85,9,56]})
```


```python
# view the dataframe
df_student_survey_data
```


```python
# check for duplicate data
df_student_survey_data.duplicated()
```


```python
# drop duplicate values with student as key
df_student_survey_data.drop_duplicates('student')
```


```python
# drop duplicate values with ID as key
df_student_survey_data.drop_duplicates('ID')
```

## Pandas SQL operations


```python
import pandas as pd
import sqlite3
```


```python
# Create SQL table
create_table = """CREATE TABLE student_score(Id INTEGER, Name VARCHAR(20),Math REAL,Science REAL);"""
```


```python
# Execute SQL statement
execute_SQL = sqlite3.connect(':memory:')
execute_SQL.execute(create_table)
execute_SQL.commit()
```


```python
# prepare SQL query
SQL_query = execute_SQL.execute('select * from student_score')
```


```python
# fetch result from sqlite database
resultset = SQL_query.fetchall()
```


```python
# view result (empty data)
resultset
```

### Prepare records in SQL using Pandas


```python
# Prepare records to be inserted into SQL table through SQL statement
insertSQL = [(10,'Jack',85,92),(29,'Tom',73,89),(65,'Ram',65.5,77),(5,'Steve',55,91)]
```


```python
# Insert records through SQL statement into SQL table
insert_statement = "Insert into student_score values (?,?,?,?)"
execute_SQL.executemany(insert_statement, insertSQL)
```


```python
# PRepare SQL query
SQL_query = execute_SQL.execute("select * from student_score")
```


```python
# Fetch resultant for the query
resultset = SQL_query.fetchall()
```


```python
# View the resultant
resultset
```


```python
# Put records into a dataframe
df_student_records = pd.DataFrame(resultset, columns=list(zip(*SQL_query.description))[0])
```


```python
# view data in dataframe
df_student_records
```
