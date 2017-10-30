---
layout: template1
---


# Introduction
In this post, we are looking at the SAT (Scholastic Aptitude Test) scores of high schoolers in the US along with other informations.
We will use <code>Python</code> for the data analysis.

The SAT is a test that high schoolers take in the US before applying to college so it is fairly important.
Moreover, SAT scores are used to indicate how good a school district is.
There have been allegations about the SAT being unfair (to certain ethnic groups in the US for example).
Doing this analysis on New York City data will help shed some light on the fairness of the SAT.

# Pre-analysis
In order to investigate the relation between the SAT and other factors such the ethnic groups or class sizes, we need data.
In the Web site of the city of New York, we can find many data sets from which these have been selected:
- <b>SAT results</b> by school (more information and download [here](https://data.cityofnewyork.us/Education/SAT-Results/f9bf-2cp4)).
- School <b>demographics</b> (more information and download [here](https://data.cityofnewyork.us/Education/School-Demographics-and-Accountability-Snapshot-20/ihfw-zy9j)).
- <b>Class size</b> at the school level (more information and download [here](https://data.cityofnewyork.us/Education/2010-2011-Class-Size-School-level-detail/urz7-pzb3)).

In what follows, we have a look on the data sets.
Then, we clean and unify all the individual datasets into a single one in order to work with the data more easily.

# Data description
In this section, we explore retrieved the data sets.
In order to achieve this, we load them into data frames.


```python
#in order to analyse the data sets, we load them into Python data frames (pandas and numpy libraries)
import pandas as pd
import numpy as np
```


```python
#initialize the array that will contain the different data sets
data = {}
```

## SAT scores by school
This data set contains the most recent school level results for New York City on the SAT.
Results are available at the school level.

It is to be noticed that the SAT test is divided into 3 sections, each of which is scored out of 800 points (last 3 columns).
The total score is out of 2400.

<b>Full data</b> can be downloaded <a href="{{ site.baseurl }}/dev/us_sat/data/SAT_Results.csv">here</a> in CSV format.



```python
#read csv file and show the data head
data["sat_results"] = pd.read_csv("data/{0}.csv".format("SAT_Results"))
data["sat_results"].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DBN</th>
      <th>SCHOOL NAME</th>
      <th>Num of SAT Test Takers</th>
      <th>SAT Critical Reading Avg. Score</th>
      <th>SAT Math Avg. Score</th>
      <th>SAT Writing Avg. Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01M292</td>
      <td>HENRY STREET SCHOOL FOR INTERNATIONAL STUDIES</td>
      <td>29</td>
      <td>355</td>
      <td>404</td>
      <td>363</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01M448</td>
      <td>UNIVERSITY NEIGHBORHOOD HIGH SCHOOL</td>
      <td>91</td>
      <td>383</td>
      <td>423</td>
      <td>366</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01M450</td>
      <td>EAST SIDE COMMUNITY SCHOOL</td>
      <td>70</td>
      <td>377</td>
      <td>402</td>
      <td>370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01M458</td>
      <td>FORSYTH SATELLITE ACADEMY</td>
      <td>7</td>
      <td>414</td>
      <td>401</td>
      <td>359</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01M509</td>
      <td>MARTA VALLE HIGH SCHOOL</td>
      <td>44</td>
      <td>390</td>
      <td>433</td>
      <td>384</td>
    </tr>
  </tbody>
</table>
</div>



## School demographics
This data set contains the annual school accounts of NYC public school student populations served by several information from which ethnicity and gender.

<b>Full data</b> can be downloaded <a href="{{ site.baseurl }}/dev/us_sat/data/Demographics_and_Accountability.csv">here</a> in CSV format.


```python
#read csv file and show the data head
data["demographics"] = pd.read_csv("data/{0}.csv".format("Demographics_and_Accountability"))
data["demographics"].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DBN</th>
      <th>Name</th>
      <th>schoolyear</th>
      <th>fl_percent</th>
      <th>frl_percent</th>
      <th>total_enrollment</th>
      <th>prek</th>
      <th>k</th>
      <th>grade1</th>
      <th>grade2</th>
      <th>...</th>
      <th>black_num</th>
      <th>black_per</th>
      <th>hispanic_num</th>
      <th>hispanic_per</th>
      <th>white_num</th>
      <th>white_per</th>
      <th>male_num</th>
      <th>male_per</th>
      <th>female_num</th>
      <th>female_per</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01M015</td>
      <td>P.S. 015 ROBERTO CLEMENTE</td>
      <td>20052006</td>
      <td>89.4</td>
      <td>NaN</td>
      <td>281</td>
      <td>15</td>
      <td>36</td>
      <td>40</td>
      <td>33</td>
      <td>...</td>
      <td>74</td>
      <td>26.3</td>
      <td>189</td>
      <td>67.3</td>
      <td>5</td>
      <td>1.8</td>
      <td>158.0</td>
      <td>56.2</td>
      <td>123.0</td>
      <td>43.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01M015</td>
      <td>P.S. 015 ROBERTO CLEMENTE</td>
      <td>20062007</td>
      <td>89.4</td>
      <td>NaN</td>
      <td>243</td>
      <td>15</td>
      <td>29</td>
      <td>39</td>
      <td>38</td>
      <td>...</td>
      <td>68</td>
      <td>28.0</td>
      <td>153</td>
      <td>63.0</td>
      <td>4</td>
      <td>1.6</td>
      <td>140.0</td>
      <td>57.6</td>
      <td>103.0</td>
      <td>42.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01M015</td>
      <td>P.S. 015 ROBERTO CLEMENTE</td>
      <td>20072008</td>
      <td>89.4</td>
      <td>NaN</td>
      <td>261</td>
      <td>18</td>
      <td>43</td>
      <td>39</td>
      <td>36</td>
      <td>...</td>
      <td>77</td>
      <td>29.5</td>
      <td>157</td>
      <td>60.2</td>
      <td>7</td>
      <td>2.7</td>
      <td>143.0</td>
      <td>54.8</td>
      <td>118.0</td>
      <td>45.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01M015</td>
      <td>P.S. 015 ROBERTO CLEMENTE</td>
      <td>20082009</td>
      <td>89.4</td>
      <td>NaN</td>
      <td>252</td>
      <td>17</td>
      <td>37</td>
      <td>44</td>
      <td>32</td>
      <td>...</td>
      <td>75</td>
      <td>29.8</td>
      <td>149</td>
      <td>59.1</td>
      <td>7</td>
      <td>2.8</td>
      <td>149.0</td>
      <td>59.1</td>
      <td>103.0</td>
      <td>40.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01M015</td>
      <td>P.S. 015 ROBERTO CLEMENTE</td>
      <td>20092010</td>
      <td></td>
      <td>96.5</td>
      <td>208</td>
      <td>16</td>
      <td>40</td>
      <td>28</td>
      <td>32</td>
      <td>...</td>
      <td>67</td>
      <td>32.2</td>
      <td>118</td>
      <td>56.7</td>
      <td>6</td>
      <td>2.9</td>
      <td>124.0</td>
      <td>59.6</td>
      <td>84.0</td>
      <td>40.4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



## Class size on the school level
This data set contains the average class sizes for each school, by grade and program type (full description [here](https://data.cityofnewyork.us/Education/2010-2011-Class-Size-School-level-detail/urz7-pzb3)).

<b>Full data</b> can be downloaded <a href="{{ site.baseurl }}/dev/us_sat/data/Class_Size-School_level_detail.csv">here</a> in CSV format.


```python
#read csv file and show the data head
data["class_size"] = pd.read_csv("data/{0}.csv".format("Class_Size-School_level_detail"))
data["class_size"].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CSD</th>
      <th>BOROUGH</th>
      <th>SCHOOL CODE</th>
      <th>SCHOOL NAME</th>
      <th>GRADE</th>
      <th>PROGRAM TYPE</th>
      <th>CORE SUBJECT (MS CORE and 9-12 ONLY)</th>
      <th>CORE COURSE (MS CORE and 9-12 ONLY)</th>
      <th>SERVICE CATEGORY(K-9* ONLY)</th>
      <th>NUMBER OF STUDENTS / SEATS FILLED</th>
      <th>NUMBER OF SECTIONS</th>
      <th>AVERAGE CLASS SIZE</th>
      <th>SIZE OF SMALLEST CLASS</th>
      <th>SIZE OF LARGEST CLASS</th>
      <th>DATA SOURCE</th>
      <th>SCHOOLWIDE PUPIL-TEACHER RATIO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>0K</td>
      <td>GEN ED</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>19.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>ATS</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>0K</td>
      <td>CTT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>ATS</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>01</td>
      <td>GEN ED</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>ATS</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>01</td>
      <td>CTT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>ATS</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>02</td>
      <td>GEN ED</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>ATS</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# Data cleaning
Having a consistent dataset will help us do analysis more quickly.
In order to do this, we’ll first need to find a common column to unify them on.
Looking at the output above, it appears that "DBN" (District Borough Number) might be that common column, as it appears in multiple datasets.
DBN is a unique code for each school.

However:
- the "class_size" data set don’t have a DBN field.
- there are several rows for each high school in both the "class_size" and the "demographics" data sets.
- the 3 components of the SAT scores (columns) need to be summed.

Thus, we need to clean these them first.

## Cleaning each data set
If we look to the data set "class_size" and to the "DBN" column in the other data sets, it looks like the DBN is actually a combination of "CSD" and "SCHOOL CODE".
There’s no systematized way to find insights like this in data, and it requires some exploration and playing around to figure out.
Let us create the column "DBN" now.


```python
#create the "DBN" column by combining other columns
data["class_size"]["DBN"] = data["class_size"].apply(lambda x: "{0:02d}{1}".format(x["CSD"], x["SCHOOL CODE"]), axis=1)
data["class_size"].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CSD</th>
      <th>BOROUGH</th>
      <th>SCHOOL CODE</th>
      <th>SCHOOL NAME</th>
      <th>GRADE</th>
      <th>PROGRAM TYPE</th>
      <th>CORE SUBJECT (MS CORE and 9-12 ONLY)</th>
      <th>CORE COURSE (MS CORE and 9-12 ONLY)</th>
      <th>SERVICE CATEGORY(K-9* ONLY)</th>
      <th>NUMBER OF STUDENTS / SEATS FILLED</th>
      <th>NUMBER OF SECTIONS</th>
      <th>AVERAGE CLASS SIZE</th>
      <th>SIZE OF SMALLEST CLASS</th>
      <th>SIZE OF LARGEST CLASS</th>
      <th>DATA SOURCE</th>
      <th>SCHOOLWIDE PUPIL-TEACHER RATIO</th>
      <th>DBN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>0K</td>
      <td>GEN ED</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>19.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>ATS</td>
      <td>NaN</td>
      <td>01M015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>0K</td>
      <td>CTT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>ATS</td>
      <td>NaN</td>
      <td>01M015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>01</td>
      <td>GEN ED</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>ATS</td>
      <td>NaN</td>
      <td>01M015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>01</td>
      <td>CTT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>ATS</td>
      <td>NaN</td>
      <td>01M015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>M</td>
      <td>M015</td>
      <td>P.S. 015 Roberto Clemente</td>
      <td>02</td>
      <td>GEN ED</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>ATS</td>
      <td>NaN</td>
      <td>01M015</td>
    </tr>
  </tbody>
</table>
</div>



In order to combine these datasets, we’ll need to find a way to condense data sets like "class_size" to the point where there’s only a single row per high school.
If not, there won’t be a way to compare SAT scores to class size.
We can accomplish this by first understanding the data better, then by doing some aggregation.
With the class_size dataset, it looks like GRADE and PROGRAM TYPE have multiple values for each school.
By restricting each field to a single value, we can filter most of the duplicate rows.



```python
class_size_temp = data["class_size"]

#restrict the class_size data set to a specific grade and program type
class_size_temp = class_size_temp[class_size_temp["GRADE "] == "09-12"]
class_size_temp = class_size_temp[class_size_temp["PROGRAM TYPE"] == "GEN ED"]

#there still are duplicates, we group the class_size data set by DBN, and take the average of each column
class_size_temp = class_size_temp.groupby("DBN").agg(np.mean)

#reset the index, so DBN is added back in as a column (otherwise, DBN is the index column)
class_size_temp.reset_index(inplace=True)

data["class_size_ge_09-12"] = class_size_temp
data["class_size_ge_09-12"].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DBN</th>
      <th>CSD</th>
      <th>NUMBER OF STUDENTS / SEATS FILLED</th>
      <th>NUMBER OF SECTIONS</th>
      <th>AVERAGE CLASS SIZE</th>
      <th>SIZE OF SMALLEST CLASS</th>
      <th>SIZE OF LARGEST CLASS</th>
      <th>SCHOOLWIDE PUPIL-TEACHER RATIO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01M292</td>
      <td>1</td>
      <td>88.0000</td>
      <td>4.000000</td>
      <td>22.564286</td>
      <td>18.50</td>
      <td>26.571429</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01M332</td>
      <td>1</td>
      <td>46.0000</td>
      <td>2.000000</td>
      <td>22.000000</td>
      <td>21.00</td>
      <td>23.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01M378</td>
      <td>1</td>
      <td>33.0000</td>
      <td>1.000000</td>
      <td>33.000000</td>
      <td>33.00</td>
      <td>33.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01M448</td>
      <td>1</td>
      <td>105.6875</td>
      <td>4.750000</td>
      <td>22.231250</td>
      <td>18.25</td>
      <td>27.062500</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01M450</td>
      <td>1</td>
      <td>57.6000</td>
      <td>2.733333</td>
      <td>21.200000</td>
      <td>19.40</td>
      <td>22.866667</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Next, we’ll need to condense the demographics dataset.
The data was collected for multiple years for the same schools.
We will only pick rows where the schoolyear field is the most recent available.


```python
#select only one year
demographics_temp = data["demographics"]
demographics_temp = demographics_temp[demographics_temp["schoolyear"] == 20112012]

data["demographics_2011-2012"] = demographics_temp
data["demographics_2011-2012"].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DBN</th>
      <th>Name</th>
      <th>schoolyear</th>
      <th>fl_percent</th>
      <th>frl_percent</th>
      <th>total_enrollment</th>
      <th>prek</th>
      <th>k</th>
      <th>grade1</th>
      <th>grade2</th>
      <th>...</th>
      <th>black_num</th>
      <th>black_per</th>
      <th>hispanic_num</th>
      <th>hispanic_per</th>
      <th>white_num</th>
      <th>white_per</th>
      <th>male_num</th>
      <th>male_per</th>
      <th>female_num</th>
      <th>female_per</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>01M015</td>
      <td>P.S. 015 ROBERTO CLEMENTE</td>
      <td>20112012</td>
      <td>NaN</td>
      <td>89.4</td>
      <td>189</td>
      <td>13</td>
      <td>31</td>
      <td>35</td>
      <td>28</td>
      <td>...</td>
      <td>63</td>
      <td>33.3</td>
      <td>109</td>
      <td>57.7</td>
      <td>4</td>
      <td>2.1</td>
      <td>97.0</td>
      <td>51.3</td>
      <td>92.0</td>
      <td>48.7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>01M019</td>
      <td>P.S. 019 ASHER LEVY</td>
      <td>20112012</td>
      <td>NaN</td>
      <td>61.5</td>
      <td>328</td>
      <td>32</td>
      <td>46</td>
      <td>52</td>
      <td>54</td>
      <td>...</td>
      <td>81</td>
      <td>24.7</td>
      <td>158</td>
      <td>48.2</td>
      <td>28</td>
      <td>8.5</td>
      <td>147.0</td>
      <td>44.8</td>
      <td>181.0</td>
      <td>55.2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>01M020</td>
      <td>PS 020 ANNA SILVER</td>
      <td>20112012</td>
      <td>NaN</td>
      <td>92.5</td>
      <td>626</td>
      <td>52</td>
      <td>102</td>
      <td>121</td>
      <td>87</td>
      <td>...</td>
      <td>55</td>
      <td>8.8</td>
      <td>357</td>
      <td>57.0</td>
      <td>16</td>
      <td>2.6</td>
      <td>330.0</td>
      <td>52.7</td>
      <td>296.0</td>
      <td>47.3</td>
    </tr>
    <tr>
      <th>27</th>
      <td>01M034</td>
      <td>PS 034 FRANKLIN D ROOSEVELT</td>
      <td>20112012</td>
      <td>NaN</td>
      <td>99.7</td>
      <td>401</td>
      <td>14</td>
      <td>34</td>
      <td>38</td>
      <td>36</td>
      <td>...</td>
      <td>90</td>
      <td>22.4</td>
      <td>275</td>
      <td>68.6</td>
      <td>8</td>
      <td>2.0</td>
      <td>204.0</td>
      <td>50.9</td>
      <td>197.0</td>
      <td>49.1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>01M063</td>
      <td>PS 063 WILLIAM MCKINLEY</td>
      <td>20112012</td>
      <td>NaN</td>
      <td>78.9</td>
      <td>176</td>
      <td>18</td>
      <td>20</td>
      <td>30</td>
      <td>21</td>
      <td>...</td>
      <td>41</td>
      <td>23.3</td>
      <td>110</td>
      <td>62.5</td>
      <td>15</td>
      <td>8.5</td>
      <td>97.0</td>
      <td>55.1</td>
      <td>79.0</td>
      <td>44.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



Now, we compute the total SAT scores from their individual 3 components.



```python
#convert each of the SAT score columns from a string to a number
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors='coerce')

#add together all of the columns to get the sat_score column, which is the total SAT score
data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

data['sat_results'].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DBN</th>
      <th>SCHOOL NAME</th>
      <th>Num of SAT Test Takers</th>
      <th>SAT Critical Reading Avg. Score</th>
      <th>SAT Math Avg. Score</th>
      <th>SAT Writing Avg. Score</th>
      <th>sat_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01M292</td>
      <td>HENRY STREET SCHOOL FOR INTERNATIONAL STUDIES</td>
      <td>29</td>
      <td>355.0</td>
      <td>404.0</td>
      <td>363.0</td>
      <td>1122.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01M448</td>
      <td>UNIVERSITY NEIGHBORHOOD HIGH SCHOOL</td>
      <td>91</td>
      <td>383.0</td>
      <td>423.0</td>
      <td>366.0</td>
      <td>1172.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01M450</td>
      <td>EAST SIDE COMMUNITY SCHOOL</td>
      <td>70</td>
      <td>377.0</td>
      <td>402.0</td>
      <td>370.0</td>
      <td>1149.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01M458</td>
      <td>FORSYTH SATELLITE ACADEMY</td>
      <td>7</td>
      <td>414.0</td>
      <td>401.0</td>
      <td>359.0</td>
      <td>1174.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01M509</td>
      <td>MARTA VALLE HIGH SCHOOL</td>
      <td>44</td>
      <td>390.0</td>
      <td>433.0</td>
      <td>384.0</td>
      <td>1207.0</td>
    </tr>
  </tbody>
</table>
</div>



## Combining the data sets
We can now combine all the data sets.
In this analysis, we ignore missing value.


```python
#joining data sets
data_full = data['sat_results']
data_full = data_full.merge(data["class_size_ge_09-12"], on="DBN", how="inner")
data_full = data_full.merge(data["demographics_2011-2012"], on="DBN", how="inner")

#the structure of the global data set
data_full.shape
```




    (434, 51)



# Data analysis
A good way to explore a dataset and see what columns are related to the one you care about is to compute correlations. This will tell you which columns are closely related to the column you’re interested in:
- The closer to 0 the correlation, the weaker the connection.
- The closer to 1, the stronger the positive correlation, and the closer to -1, the stronger the negative correlation.


```python
#show the correlations of the columns with the "sat_score" column
data_full_corr = data_full.corr()['sat_score']
data_full_corr.sort_values(ascending=False)
```




    sat_score                            1.000000
    SAT Writing Avg. Score               0.981751
    SAT Critical Reading Avg. Score      0.976819
    SAT Math Avg. Score                  0.956406
    white_per                            0.646568
    asian_per                            0.552204
    asian_num                            0.483939
    white_num                            0.460505
    female_num                           0.403581
    NUMBER OF STUDENTS / SEATS FILLED    0.400095
    AVERAGE CLASS SIZE                   0.395964
    total_enrollment                     0.385741
    NUMBER OF SECTIONS                   0.364537
    male_num                             0.345808
    SIZE OF LARGEST CLASS                0.327099
    SIZE OF SMALLEST CLASS               0.282388
    female_per                           0.107676
    sped_num                             0.058782
    CSD                                  0.054011
    hispanic_num                         0.052672
    black_num                            0.043408
    male_per                            -0.107624
    ell_num                             -0.130355
    black_per                           -0.302794
    hispanic_per                        -0.363696
    ell_percent                         -0.379510
    sped_percent                        -0.420646
    frl_percent                         -0.718163
    SCHOOLWIDE PUPIL-TEACHER RATIO            NaN
    schoolyear                                NaN
    Name: sat_score, dtype: float64



This gives us a few insights that can be explored. For example:
- The percentage of females at a school (female_per) correlates positively with SAT score, whereas the percentage of males (male_per) correlates negatively.
- There is a significant racial inequality in SAT scores (white_per, asian_per, black_per, hispanic_per).

Each of these items is a potential angle to explore and tell a story about using the data.


## Exploring race and SAT scores
One angle to investigate involves race and SAT scores.
There is a large correlation difference, and plotting it out will help us see that.


```python
#importing the matplotlib library
import matplotlib.pyplot as plt
%matplotlib inline

#2D graph
data_full.corr()["sat_score"][["white_per", "asian_per", "black_per", "hispanic_per"]].plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x109172b38>




![png](output_27_1.png)


It looks like the higher percentages of white and asian students correlate with higher SAT scores, but higher percentages of black and hispanic students correlate with lower SAT scores.
For hispanic students, this may be due to the fact that there are more recent immigrants who are english learners.

## Gender differences in SAT scores
An other angle to explore is the relationship between gender and SAT score.
We noted that a higher percentage of females in a school tends to correlate with higher SAT scores.
We can visualize this with a bar graph.


```python
#2D graph
data_full.corr()["sat_score"][["male_per", "female_per"]].plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x109c5f160>




![png](output_30_1.png)


To dig more into the correlation, we can make a scatterplot of "female_per" and "sat_score".


```python
#scatter plot
data_full.plot.scatter(x='female_per', y='sat_score')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x104209978>




![png](output_32_1.png)


It looks like there’s a cluster of schools with a high percentage of females, and very high SAT scores (in the top right).
We can get the names of the schools in this cluster.


```python
data_full[(data_full["female_per"] > 65) & (data_full["sat_score"] > 1400)]["SCHOOL NAME"]
```




    8                         BARD HIGH SCHOOL EARLY COLLEGE
    27              PROFESSIONAL PERFORMING ARTS HIGH SCHOOL
    31                         ELEANOR ROOSEVELT HIGH SCHOOL
    48                          TALENT UNLIMITED HIGH SCHOOL
    78     FIORELLO H. LAGUARDIA HIGH SCHOOL OF MUSIC & A...
    364                          TOWNSEND HARRIS HIGH SCHOOL
    411         FRANK SINATRA SCHOOL OF THE ARTS HIGH SCHOOL
    Name: SCHOOL NAME, dtype: object



Searching Google reveals that these are elite schools that focus on the performing arts. These schools tend to have higher percentages of females, and higher SAT scores. This likely accounts for the correlation between higher female percentages and SAT scores, and the inverse correlation between higher male percentages and lower SAT scores.

# Conclusion

In this post, we have analysed the schools SAT scores and the relation with some other informations like ethnic groups and gender.
However, we explored some angles only in the surface, and could have dived into more.
Furthermore, we could have combined with other data sets such as schools location informations.

# Credits
This case study is based on an article of [Vik Paruchuri](https://twitter.com/vikparuchuri "@vikparuchuri").
I thank him for this example.


```python

```
