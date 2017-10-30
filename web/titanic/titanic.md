---
layout: template1
---


# Introduction
This case study is about predicting which passengers survived the [sinking of the famous Titanic](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic). 
In our work, we would like to establish a model that predicts the survival of each passenger. In order to do this, we will use a dataset that describe each passenger (multiple features) and if they survived or not. 

# Data description
In this section, we load and explore the dataset. First, we import the libraries needed along the project.


```python
# Import numerical and data processing libraries
import numpy as np
import pandas as pd

# Import helpers that make it easy to do cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Import visualisation libraries
import matplotlib.pyplot as plt
%matplotlib inline

# Import a method in order to make deep copies
from copy import deepcopy

# Import an other usefull libraries
import itertools

# Set the paths for inputs and outputs
local = 1
if(local == 0):
    inputPath = "../input/"
    outputPath = "../output/"
else:
    inputPath = "data/"
    outputPath = "data/"
```

Here, we load the datasets. We actually have 2 datasets:
- "train.csv": contains informations about some passengers (multiple columns) and the fact that they survived or not (one column). You may download this dataset <a href="https://www.kaggle.com/c/titanic/data">here</a> in CSV format.
- "test.csv": contains informations about some other passengers (multiple columns) but without the survival information. You may download this dataset <a href="https://www.kaggle.com/c/titanic/data">here</a> in CSV format.

Note that the only difference in the structure of the 2 datasets is that the "test" dataset does not contain "Survived" column (the "label" or the "class" to which the passenger belongs).

We describe in what follows the columns of the "train" dataset.


```python
# This creates a pandas dataframe and assigns it to the titanic variable
titanicOrigTrainDS = pd.read_csv(inputPath + "train.csv")
titanicTrainDS = deepcopy(titanicOrigTrainDS)

titanicOrigTestDS = pd.read_csv(inputPath + "test.csv")
titanicTestDS = deepcopy(titanicOrigTestDS)

# Print the first five rows of the dataframe
titanicTrainDS.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Here is a short description of the different columns:
- <b>PassengerId</b>: Id of the passenger
- <b>Pclass</b>: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- <b>Name</b>: Name	
- <b>Sex</b>: Sex	
- <b>Age</b>: Age in years	
- <b>Sibsp</b>: Number of siblings / spouses aboard the Titanic	
- <b>Parch</b>: Number of parents / children aboard the Titanic	
- <b>Ticket</b>: Ticket number	
- <b>Fare</b>: Passenger fare	
- <b>Cabin</b>: Cabin number	
- <b>Embarked</b>: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- <b>Survival</b>: Survival (0 = No, 1 = Yes)


# Hypothesis

Let's consider which variables might affect the outcome of survival (feature selection).
In this section, we test the variability of the survival percentage according to each feature.
It is to be noted that a variability induce that the feature has some influence. 
But the opposite is not automatically true.

We consider the following features:
- "Pclass": knowing that first class cabins were closer to the deck of the ship, are passengers from the first class more likely to survive? If yes, passenger class "Pclass" might affect the outcome.
- "Fare": the passenger fare is probably tied to passenger class and could have a correlation too.
- "Sex": are women more likely to survive? If yes, "Sex" would be a good predictor.
- "Age": are children more likely to survive? If yes, "Age" would be a good predictor.
- "Embarked": people who boarded at certain ports may have had cabins closer or farther away exits.
- "Sibsp": does being alone give more chance of surviving or less because no one is thinking about you.
- "Parch": same remark.

However, we do not consider these features:
- PassengerId
- Name
- Ticket
- Cabin

Let us explore the pre-selected features and their correlations with the variable "Survived".






```python
# What is the percentage of survival by class (1st, 2nd, 3rd)?
titanicTrainDS[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() 

# We find a big variability. The first class passengers had definetely more chances to survive. 
# This means that "Pclass" is an important feature.
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the percentage of survival by sex?
titanicTrainDS[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()

# We find a huge variability. Woman had more chances to survive. 
# This is definitely an important feature.
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the percentage of survival according to the port of embarkation
titanicTrainDS[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>0.389610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>0.336957</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the percentage of survival according to the number of siblings?
titanicTrainDS[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.345395</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.535885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.464286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the percentage of survival according to the number of parents?
titanicTrainDS[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Parch</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.343658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.550847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the percentage of survival according to the age (grouped)?
interval = 10
TempV = round(titanicTrainDS["Age"]//interval)*interval
titanicTrainDS["AgeIntervalMin"] = TempV
titanicTrainDS["AgeIntervalMax"] = TempV + interval
titanicTrainDS[["AgeIntervalMin", "AgeIntervalMax", "Survived"]].groupby(["AgeIntervalMin"], as_index=False).mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AgeIntervalMin</th>
      <th>AgeIntervalMax</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.612903</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>20.0</td>
      <td>0.401961</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>30.0</td>
      <td>0.350000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30.0</td>
      <td>40.0</td>
      <td>0.437126</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.0</td>
      <td>50.0</td>
      <td>0.382022</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50.0</td>
      <td>60.0</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>60.0</td>
      <td>70.0</td>
      <td>0.315789</td>
    </tr>
    <tr>
      <th>7</th>
      <td>70.0</td>
      <td>80.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>80.0</td>
      <td>90.0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the percentage of survival according to the fare (grouped)?
interval = 25
TempV = round(titanicTrainDS["Fare"]//interval)*interval
titanicTrainDS["FareIntervalMin"] = TempV
titanicTrainDS["FareIntervalMax"] = TempV + interval
titanicTrainDS[["FareIntervalMin", "FareIntervalMax", "Survived"]].groupby(["FareIntervalMin"], as_index=False).mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FareIntervalMin</th>
      <th>FareIntervalMax</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>25.0</td>
      <td>0.287253</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.0</td>
      <td>50.0</td>
      <td>0.421965</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50.0</td>
      <td>75.0</td>
      <td>0.546875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75.0</td>
      <td>100.0</td>
      <td>0.795455</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.0</td>
      <td>125.0</td>
      <td>0.733333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>125.0</td>
      <td>150.0</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>6</th>
      <td>150.0</td>
      <td>175.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200.0</td>
      <td>225.0</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>225.0</td>
      <td>250.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250.0</td>
      <td>275.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500.0</td>
      <td>525.0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We decide to keep all pre-selected features.
However, some of them need to be "cleaned" before running models on our datasets.

# Data cleaning
Let us have a look to the datasets ("train" and "test").
Note that we need to do the cleaning part in parallel for both datasets.


```python
titanicDSs = [titanicTrainDS, titanicTestDS]
```


```python
# lenght of the dataframe
len(titanicTrainDS)
```




    891




```python
# Summary on the dataframe
titanicTrainDS.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>AgeIntervalMin</th>
      <th>AgeIntervalMax</th>
      <th>FareIntervalMin</th>
      <th>FareIntervalMax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
      <td>25.252101</td>
      <td>35.252101</td>
      <td>22.615039</td>
      <td>47.615039</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
      <td>14.970969</td>
      <td>14.970969</td>
      <td>49.512306</td>
      <td>49.512306</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
      <td>20.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
      <td>20.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>30.000000</td>
      <td>40.000000</td>
      <td>25.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
      <td>80.000000</td>
      <td>90.000000</td>
      <td>500.000000</td>
      <td>525.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# lenght of the dataframe
len(titanicTestDS)
```




    418




```python
# Summary on the dataframe
titanicTestDS.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>332.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>2.265550</td>
      <td>30.272590</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>35.627188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.841838</td>
      <td>14.181209</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>55.907576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



If we have a look to the first dataset (the "train" one), we see that all the numerical columns have indeed a count of 891 except the "Age" column that has a count of 714. 
This indicates that there are missing values (null, NA, or not a number).

As we don't want to remove the rows with missing values, we choose to clean the data by filling in all of the missing values.
It would be a good idea to test if the missing value for "Age" is correlated with other variable. 
For example, we see that it is there are way more missing values for the "Q" port of embarkation.


```python
titanicTrainDS["AgeEmptyOrNot"] =  titanicTrainDS["Age"].apply(lambda x: 1 if x>=0  else 0)
titanicTrainDS[['Embarked', 'AgeEmptyOrNot']].groupby(['Embarked'], as_index=False).mean() 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>AgeEmptyOrNot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>0.773810</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>0.363636</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>0.860248</td>
    </tr>
  </tbody>
</table>
</div>



However, the mean age does not seem to differ strongly according to the port of embarkation.


```python
titanicTrainDS[['Embarked', 'Age']].groupby(['Embarked'], as_index=False).mean() 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>30.814769</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>28.089286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>29.445397</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we decide to clean the data by filling in all of the missing values with simply the median of all the values in the column


```python
# Fill missing values with the median value
for dataset in titanicDSs:
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
```

The "Sex" column is non-numeric, we need to convert it.
But first, we confirm that this column does not have empty values. then we make the conversion.


```python
# What are the values for this column?
for dataset in titanicDSs:
    print(dataset["Sex"].unique())
```

    ['male' 'female']
    ['male' 'female']



```python
# Convert to numerical values
for dataset in titanicDSs:
    dataset.loc[dataset["Sex"] == "male", "Sex"] = 0
    dataset.loc[dataset["Sex"] == "female", "Sex"] = 1
```

We do the same with the "Embarked" column. 
We First analyse if there are missing values. 
We will see that yes and choose to fill the missing values with the most frequent value.


```python
# What are the values for this column?
for dataset in titanicDSs:
    print(dataset["Embarked"].unique())
```

    ['S' 'C' 'Q' nan]
    ['Q' 'S' 'C']



```python
# Fill missing values with most frequent value
mostFrequentOccurrence = titanicTrainDS["Embarked"].dropna().mode()[0]
titanicTrainDS["Embarked"] = titanicTrainDS["Embarked"].fillna(mostFrequentOccurrence)

# Convert to numerical values
for dataset in titanicDSs:
    dataset.loc[dataset["Embarked"] == "S", "Embarked"] = 0
    dataset.loc[dataset["Embarked"] == "C", "Embarked"] = 1
    dataset.loc[dataset["Embarked"] == "Q", "Embarked"] = 2
```

Finally, we clean the "Fare" variable of the "test" dataset.


```python
titanicTestDS["Fare"] = titanicTestDS["Fare"].fillna(titanicTestDS["Fare"].median())
```

# Model application
Now, we can turn to the core of the analysis.
We will introduce a couple of functions.
The first function is the one that will enable evaluating the accuracy of one classification method type.
However, we introduce a second function that enables to run the first function on each combination of predictors (ex: ["Sex", "Age", "Embarked"] or ["Age", "SibSp", "Parch", "Fare"] etc.).

In what follows, we build the list of combinations and then introduce the these functions.


```python
# The columns that can be used in the prediction
predictorsAll = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] 

# Create all combinations of predictors
predictorCombinations = [] # all combination of predictord
for index in range(1, len(predictorsAll)+1):
    for subset in itertools.combinations(predictorsAll, index):
         predictorCombinations.append(list(subset))  
            
#predictorCombinations
```


```python
# Function: Evaluate one algorithm type (and return n fitted algorithms)
# -input
    # predictorsDs: the dataset projected to the predictors of interest
    # targetDs: the target or label vector of interest (the column "Survived" in our work)
    # algModel: the "template" or model of the algorithm to apply
    # nbFK: the number of cross validation folders
# -output
    # algs: nbKF fitted algorithms 
    # accuracy: the evaluation of the accuracy
def binClassifModel_kf(predictorsDs, targetDs, algModel, nbKF):
    # List of algorithms
    algs = []
    
    # Generate cross-validation folds for the titanic data set
    # It returns the row indices corresponding to train and test
    # We set random_state to ensure we get the same splits every time we run this
    kf = KFold(nbKF, random_state=1)

    # List of predictions
    predictions = []

    for trainIndexes, testIndexes in kf.split(predictorsDs):
        # The predictors we're using to train the algorithm  
        # Note how we only take the rows in the train folds
        predictorsTrainDs = (predictorsDs.iloc[trainIndexes,:])
        # The target we're using to train the algorithm
        train_target = targetDs.iloc[trainIndexes]
        
        # Initialize our algorithm class
        alg = deepcopy(algModel)
        # Training the algorithm using the predictors and target
        alg.fit(predictorsTrainDs, train_target)
        algs.append(alg)
        
        # We can now make predictions on the test fold
        thisSlitpredictions = alg.predict(predictorsDs.iloc[testIndexes,:])
        predictions.append(thisSlitpredictions)


    # The predictions are in three separate NumPy arrays  
    # Concatenate them into a single array, along the axis 0 (the only 1 axis) 
    predictions = np.concatenate(predictions, axis=0)

    # Map predictions to outcomes (the only possible outcomes are 1 and 0)
    predictions[predictions > .5] = 1
    predictions[predictions <=.5] = 0
    accuracy = len(predictions[predictions == targetDs]) / len(predictions)
    
    # return the multiple algoriths and the accuracy
    return [algs, accuracy]
```


```python
# Helper that return the indexed of the sorted list
def sort_list(myList):
    return sorted(range(len(myList)), key=lambda i:myList[i])

# Function: Run multiple evaluations for one algorithm type (one for each combination of predictors)
# -input
    # algModel: the "template" or model of the algorithm to apply
    # nbFK: the number of cross validation folders
# -output
    # {}
def getAccuracy_forEachPredictor(algModel, nbKF):
    accuracyList = []
    
    # For each combination of predictors
    for combination in predictorCombinations:
        result = binClassifModel_kf(titanicTrainDS[combination], titanicTrainDS["Survived"], algModel, nbKF)
        accuracy = result[1]
        accuracyList.append(accuracy)

    # Sort the accuracies
    accuracySortedList = sort_list(accuracyList)

    # Diplay the best combinations
    for i in range(-5, 0):
        print(predictorCombinations[accuracySortedList[i]], ": ", accuracyList[accuracySortedList[i]])
    #for elementIndex in sort_list(accuracyList1):
    #    print(predictorCombinations[elementIndex], ": ", accuracyList1[elementIndex])
        
    print("--------------------------------------------------")

    # Display the accuracy corresponding to combination that uses all the predictors
    lastIndex = len(predictorCombinations)-1
    print(predictorCombinations[lastIndex], ":", accuracyList[lastIndex])
```

Now that we have introduce the above functions, we evaluate a set of classification methods on each combination of predictors.
Here are the evaluated classification methods:
- LinearRegression
- LogisticRegression
- GaussianNB
- KNeighborsClassifier
- DecisionTreeClassifier
- RandomForestClassifier


```python
algModel = LinearRegression(fit_intercept=True, normalize=True)
getAccuracy_forEachPredictor(algModel, 5)
```

    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/scipy/linalg/basic.py:884: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
      warnings.warn(mesg, RuntimeWarning)


    ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'] :  0.7912457912457912
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'] :  0.792368125701459
    ['Pclass', 'Sex', 'Age', 'SibSp'] :  0.7934904601571269
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'] :  0.7934904601571269
    ['Pclass', 'Sex', 'Age', 'Parch', 'Embarked'] :  0.7934904601571269
    --------------------------------------------------
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] : 0.7878787878787878



```python
algModel = LogisticRegression()
getAccuracy_forEachPredictor(algModel, 5)
```

    ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'] :  0.7946127946127947
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked'] :  0.7946127946127947
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] :  0.7946127946127947
    ['Pclass', 'Sex', 'SibSp', 'Parch'] :  0.7968574635241302
    ['Pclass', 'Sex', 'SibSp'] :  0.7991021324354658
    --------------------------------------------------
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] : 0.7946127946127947



```python
algModel = GaussianNB()
getAccuracy_forEachPredictor(algModel, 5)
```

    ['Sex', 'SibSp', 'Parch', 'Embarked'] :  0.7957351290684624
    ['Sex', 'Age', 'SibSp', 'Fare', 'Embarked'] :  0.7957351290684624
    ['Sex', 'SibSp'] :  0.7968574635241302
    ['Sex', 'SibSp', 'Embarked'] :  0.7968574635241302
    ['Sex', 'SibSp', 'Parch'] :  0.797979797979798
    --------------------------------------------------
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] : 0.7856341189674523



```python
algModel = KNeighborsClassifier(n_neighbors=5)
getAccuracy_forEachPredictor(algModel, 5)
```

    ['Pclass', 'Sex', 'Parch', 'Embarked'] :  0.7878787878787878
    ['Sex', 'SibSp', 'Embarked'] :  0.7890011223344556
    ['Pclass', 'Sex'] :  0.7901234567901234
    ['Pclass', 'Sex', 'Age', 'SibSp'] :  0.7912457912457912
    ['Sex', 'SibSp', 'Parch'] :  0.8002244668911336
    --------------------------------------------------
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] : 0.691358024691358



```python
algModel = DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=2)
getAccuracy_forEachPredictor(algModel, 5)
```

    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'] :  0.8047138047138047
    ['Pclass', 'Sex', 'Parch', 'Embarked'] :  0.8058361391694725
    ['Pclass', 'Sex', 'Parch', 'Fare', 'Embarked'] :  0.8058361391694725
    ['Pclass', 'Sex', 'Fare', 'Embarked'] :  0.8103254769921436
    ['Pclass', 'Sex', 'Embarked'] :  0.8114478114478114
    --------------------------------------------------
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] : 0.797979797979798



```python
algModel = RandomForestClassifier(n_estimators=100, min_samples_split=4, min_samples_leaf=2)
getAccuracy_forEachPredictor(algModel, 5)
```

    ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked'] :  0.8249158249158249
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] :  0.8282828282828283
    ['Pclass', 'Sex', 'Age', 'Fare'] :  0.8305274971941639
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'] :  0.8305274971941639
    ['Pclass', 'Sex', 'Age', 'Parch', 'Fare'] :  0.835016835016835
    --------------------------------------------------
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] : 0.8282828282828283


# Prediction application
After having run all the models, we decide to choose the model that gave the best performance.
This model is the "RandomForestClassifier" with the specific parameters above.
Furthermore, we will use it with the best combination of predictors which is ['Pclass', 'Sex', 'Age', 'Parch', 'Fare'] that gave approximately 83% of accuracy.


```python
# Run again the model with the tuned parameters on the dataset using the best combination of predictors
algModel = RandomForestClassifier(n_estimators=100, min_samples_split=4, min_samples_leaf=2)
predictors = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
result = binClassifModel_kf(titanicTrainDS[predictors], titanicTrainDS["Survived"], algModel, 5)
algList = result[0] # the set of algorithms

predictionsList = []
for alg in algList:
    predictions = alg.predict(titanicTestDS[predictors])
    predictionsList.append(predictions)    

# There are different preditions, we take the mean (a voting-like system)
predictionsFinal = np.mean(predictionsList, axis=0)

# Map predictions to outcomes (the only possible outcomes are 1 and 0)
predictionsFinal[predictionsFinal > .5] = 1
predictionsFinal[predictionsFinal <=.5] = 0

# Cast as int
predictionsFinal = predictionsFinal.astype(int)
```

Finally, we can generate the submission file with the prediction of the survival for each passenger of the test dataset.


```python
# Create a new dataset with only the id and the target column
submission = pd.DataFrame({
        "PassengerId": titanicTestDS["PassengerId"],
        "Survived": predictionsFinal
    })

#submission.to_csv(outputPath + 'submission.csv', index=False)
```

# Conclusion

Throughout this work, we tried to establish a good model for predicting the survival of passengers in the Titanic disaster.
As outlooks, we could investigate the influence of some features cleaning and scaling (such as the "Fare" scaling) on the overall performance.

# References
Many ideas in this work are inspired by the great [tutorials](https://www.kaggle.com/c/titanic#tutorials) of the Titanic competition and other sources.


```python

```
