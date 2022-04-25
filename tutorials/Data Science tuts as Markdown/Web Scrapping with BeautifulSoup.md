## Web document, parser and Object


```python
# import bs4 the beautifulSoup library
from bs4 import BeautifulSoup

# create a html document
html_doc = """<html>
<body>
<h1>First Heading</h1>
<b><!-- This is comment --></b>
<p title="About Me" class="test">My first paragraph.</p>
<div class="cities">
<h2>London</h2>
</div>
</body>
</html>
"""

# parse it using html parser
soup = BeautifulSoup(html_doc, 'html.parser')
```


```python
# view the soup type
type(soup)
```


```python
# view the soup object
print(soup)
```


```python
# create a tag object
tag = soup.p
```


```python
# view the tag object type
type(tag)
```


```python
# print the tag
print(tag)
```


```python
# create comment object type

```


```python
# create the comment object type
comment = soup.b.string
```


```python
# view the comment type
type(comment)
```


```python
# view the comment
print(comment)
```


```python
# view tag attributes
tag.attrs
```


```python
# view tag value
tag.string
```


```python
# view tag type(navigable string)
type(tag.string)
```

### Search the tree with filters


```python
# import the required libraries
from bs4 import BeautifulSoup
```


```python
# import the web scrapping example html file
# change as per path of your html file
HTMLfilePath = "web_scrapping_example.html"  # change as per the path
```


```python
# view the content of the soup object
soup.contents
```


```python
# search using find methods
tag_li = soup.find("li")
```


```python
# print the tag type
type(tag_li)
```


```python
# print the tag
tag_li
```


```python
# search the document suing find mehthod for an ID
find_all = soup.find(id='HR')
```


```python
# pint the find_id object
find_id
```


```python
# print the string value
find_id.li.div.string
```


```python
# search using string value
search_for_stringOnly = soup.findAll(text=['Kelly','Jack'])
```


```python
# print the search value
search_for_stringOnly
```


```python
# search based on css classname (present as attribute)
css_class_search = soup.find(attrs={'class'='ITManager'})
css_class_search
```


```python
# create a function to search the document based upon the tag passed as parameter
def is_account_manager(tag):
    return tag.has_attr("id") and tag.get("id") == "Finance"
```


```python
# search the document using function and print it
account_manager = soup.find(is_account_manager)
account_manager.li.div.string
```


```python
# print tag name using True - which returns all the tags present in the document
for tag in soup.findAll(True):
    print(tag.name)
```


```python
# search using findall method for the given class
find_class = soup.findAll(class='HRManager')
```


```python
# view the type of class
type(find_class)
```


```python
# print the second resultant
print(find_class[0])
```


```python
# print the second result
print(find_class[1])
```


```python
# find parnts using find parent method
find_class = find_class[0]
find_parent = find_class.find_parent["ul"]
find_parent
```


```python
# now use find method to search based on the id
org = soup.find(id="IT")
```


```python
# print the search object
print(org)
```


```python
# find the next siblings
next_sibling = org.findNextSiblings()
```


```python
# print parents
parent = org.findParents
print(parent)
```


```python
# find and print previous
all_previous = org.findAllPrevious()
print(all_previous)
```


```python
# search and print previous sibling
previous_sibling = org.findPreviousSibling()

```


```python
# search and print all next
all_next = org.findAllNext()
print(all_next)
```


```python
# use regular expression to search the document
import re
email_example = """<br>
<p>my email id is:</p>
abc@exmaple.com"""
soup_email = BeautifulSoup(email_example, "lxml")

# use compile method to compile the information which contains regular expression
emailID_regex = re.compile("\w+@\w+\.\w+")

# find and print the mail id using regular expression
email_id = soup_email.find(text = emailID_regex)
print(email_id)
```

### navigating the tree


```python
# import required library
from bs4 import BeautifulSoup
```


```python
# create html document
book_html_doc = """
<catalog>
<head><title>The web catalog</title></head>
<p class="title"><b>the book catalog</b></p>
<books>
    <book id="bk001">
    <author>Hightower, Kim</author>
    <title>The first book</title>
        <genre>Fiction</genre>
        <price>44.90</price>
        <pub_date>2000-10-01</pub_date>
        <review>An interesting story of nothing</review>
    </book>
    <book id="bk002">
    <author>Nitin Kumar</author>
    <title>The second book</title>
        <genre>Biography</genre>
        <review>An masterpiece story of nothing</review>
    </book>
    <book id="bk003">
    <author>Yash Singh</author>
    <title>The third book</title>
        <genre>Poem</genre>
        <price>24.90</price>
        <review>An short poem of nothing</review>
    </book>
</books></catalog>"""
```


```python
# create soup object
book_soup = BeautifulSoup(book_html_doc, 'html.parser')
```


```python
# print catalog tag
print(book_soup.catalog)
```


```python
# view the head of the book html doc
book_soup.head
```


```python
# view the title of the book html doc
title_tag = book_soup.title
print(title_tag)
```


```python
# print the catalog bold tag
book_soup.catalog.b
```


```python
# navigate down the descendants and print sum
for descen in book_soup.head.descendants:
    print(descen)
```


```python
# navigate down using stripped string method
for string in book_soup.stripped_strings:
    print(repr(string))
```


```python
# navigate up using parent method
title_tag.parent
```


```python
# create element object to navigate back and forth
element_soup = book_soup.catalog.books
```


```python
# navigate forward using next_element method
next_element = element_soup.next_element.next_element
next_element
```


```python
# navigate back using previous_element method
previous_element = next_element.previous_element.previous_element
previous_element
```


```python
# create a sibling object and navigate to view it
next_sibling = book_soup.catalog.books.book
next_sibling
```


```python
# navigate to next sibling
next_sibling2 = next_sibling.next_sibling
next_sibling2.next_sibling
```


```python
# navigate to previous sibling
previous_sibling = next_sibling2.previous_sibling
previous_sibling
```

### Modifying the tree


```python
# import the required library
from bs4 import BeautifulSoup
```


```python
# create employee html document
employee_html_doc = """
<employees>
    <employee class="accountant">
        <firstName>John</firstName> <lastName>Doe</lastName>
    </employee>
    <employee class="manager">
        <firstName>Anna</firstName> <lastName>Smith</lastName>
    </employee>
    <employee class="developer">
        <firstName>Peter</firstName> <lastName>Jones</lastName>
    </employee>
</employees>
"""
```


```python
# create soup object
soup_emp = BeautifulSoup(employee_html_doc, 'html.parser')
```


```python
# access and view the tag
tag = soup_emp.employee
tag
```


```python
# modify the tag
tag['class'] = 'manager'
```


```python
# view the tag to see the modification
tag
```


```python
# view the soup object to verify the modification
soup_emp
```


```python
# add a tag
tag = soup_emp.new_tag('rank')
tag.string = 'Manager1'

# modify using insert after modification
soup_emp.employees.employee.insert_after(tag)
```


```python
# view the soup object
print(soup_emp)
```


```python
# clear all the modified tags
tag.clear()
```


```python
# view the soup object
soup_emp
```


```python
# create a tag object and view it
tag = soup_emp.employees.employee
tag
```


```python
# extract the information using extract method
tag.firstname.string.extract()
```


```python
tag.firstname.replace_with('first name')
```


```python
soup_emp.employees
```

### parse part of the document


```python
# import required library
from bs4 import BeautifulSoup
```


```python
# sample web document from www.simplilearn.com website
data_SL = """
<ul class="content-col_discover">
<h5>Discover</h5>
<li><a href="/resources" id="free_resources">Free Resources</a><li>
<li><a href="http://community.simplilearn.com/" id="community">Simplilearn Community</a><li>
<li><a href="/career-data-labs" id="lab">Career daa labs</a><li>
<li><a href="/scholarships-for-veterans" id="scholarship">Veteran Scholarship</a><li>
<li><a href="http://www.simplilearn.com/feed" id="rss">RSS feed</a><li>
</ul>
"""
```


```python
# create soup object
soup_SL = BeautifulSoup(data_SL, 'html.parser')
```


```python
# parse only part of document, text(string) values for tags using getText method
print(soup_SL.get_text())
```


```python
# import SoupStrainer class for parsing the desired part of the web document
from bs4 import SoupStrainer
```


```python
# create object to parse only the id(link) with lab
tags_with_lablink = SoupStrainer(id="lab")
```


```python
# print the part of the parsed document
print(BeautifulSoup(data_SL, 'html.parser', parse_only=tags_with_lablink).prettify())
```

### printing and formatting output


```python
# import the required libraries
from bs4 import BeautifulSoup
import requests
```


```python
# define url for simplilearn
url = "http://simplilearn.com"
```


```python
# access result through requests object
result = requests.get(url)

# load the page content
page_content = result.content

# create soup object
soup = BeautifulSoup(page_content, 'html.parser')
```


```python
# view the contents
soup.contents
```


```python
# pretify the output
print(soup.prettify())
```


```python
# view the original encoding
soup.original_encoding
```


```python
# format the tag a to xml
soup.body.a.prettify(formatter='xml')
```


```python
# define a custom function to convert string values to uppercase
def upperCaseFn(strtext):
    return strtext.upper()

# format using custom function for outputing string texts in uppercase
soup.body.a.prettify(formatter=upperCaseFn)
```
