### Eliminate punctuation and stopwords from the sentence


```python
# import required libraries
import string
from nltk.corpus import stopwords
```


```python
# View first 10 stopwords present in the english corpus
stopwords.words('english')[0:10]
```


```python
# for i in stopwords.words('english'):
#     if i=='fine':
#         print(i)

[i for i in stopwords.words('english') if i=='me']  #for searching 'me' inside the corpus 'english'
```


```python
# Create a test sentence
test_sentence = 'This is my first string. Wow! we are doing just fine'
```


```python
# Eliminate the punctuation in form of characters and print them
no_punctuation = [char for char in test_sentence if char not in string.punctuation]
no_punctuation
```


```python
# Now eliminate the punctuation and print them as a whole sentence
no_punctuation = ''.join(no_punctuation)
no_punctuation
```


```python
# Split each words present in the new sentence
no_punctuation.split()
```


```python
# Now eliminate stopwords
clean_sentence = [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
```


```python
# Print the final cleaned sentence
clean_sentence
```

##### eliminated strings - ~~This~~ ~~is~~ ~~my~~ ~~we~~ ~~are~~ ~~doing~~ ~~just~~


```python
# load the dataset
from sklearn.datasets import load_digits
```


```python
# create object of the dataset
digit_dataset = load_digits()
```


```python
# use built-in DESCR function to describe dataset
digit_dataset.DESCR
```


```python
type(digit_dataset)
```


```python
digit_dataset.data
```


```python
digit_dataset.target
```

## Bag of words


```python
# import required library
from sklearn.feature_extraction.text import CountVectorizer
```


```python
# instantiate the vectotizer
vectorizer = CountVectorizer()
```


```python
# create 3 documents
doc1 = "This is first document"
doc2 = "This is second document"
doc3 = "This is third document"
```


```python
# put them together
listofdocument = [doc1, doc2, doc3]
```


```python
# fit them as bag of words
bag_of_words = vectorizer.fit(listofdocument)
```


```python
# check bag of words
bag_of_words
```


```python
# apply transform method
bag_of_words = vectorizer.transform(listofdocument)
```


```python
# print bag of words
print(bag_of_words)
```


```python
# verify the vocabulary for repeated word
print(vectorizer.vocabulary_.get('second'))
print(vectorizer.vocabulary_.get('document'))
```


```python
# Check the type of bag of words
type(bag_of_words)
```

## Pipeline and grid search


```python
# import required libraries
import pandas as pd
import string
from pprint import pprint
from time import time
```


```python
# import the dataset
df_spam_collection = pd.read_csv('', sep='\t', names = ['response','message'])
```


```python
# view first 5 records with head method
df_spam_collection.head()
```


```python
# import text processing libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# import SGD classifier
from sklearn.linear_model import SGDClassifier

# import for gridsearch
from sklearn.model_selection import GridSearchCV

# import for pipeline
from sklearn.pipeline import Pipeline

# define the pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())
])
```


```python
# parameters for grid search
parameters = {'tfidf__use_idf': (True, False)}
```


```python
# perform the gridsearch with pipeline and parameters
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
print('Performing grid search now ...')
print('parameters:')
pprint(parameters)
t0 = time()
grid_search.fit(df_spam_collection['message'],df_spam_collection['response'])
print("Done in %0.3fs" %(time()-t0))
print()
```


```python

```
