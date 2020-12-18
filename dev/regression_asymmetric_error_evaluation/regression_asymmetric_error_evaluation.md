# Introduction

In this technical study, we are interested in the some types of regression problems, whose errors should be evaluated asymetrically.
As an example, imagin you sell newspappers and that the cost of production of a newspapper is 0.5€, but you can sell it 2€.
It implies that:
- for every newspapper you do not sell, you loose 0.5€
- for every newspapper that you miss selling, you loose 2€ - 0.5€ (of printing) = 1.5€

In this study, we will present and evaluate 3 different techniques to tackle this problem.




```python
import math
import pandas as pd

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import six
import sys
sys.modules['sklearn.externals.six'] = six
from skgarden import RandomForestQuantileRegressor

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)

RANDOM_STATE = 42
```

    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.ensemble.forest module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.ensemble. Anything that cannot be imported from sklearn.ensemble is now part of the private API.
      warnings.warn(message, FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.tree.tree module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.tree. Anything that cannot be imported from sklearn.tree is now part of the private API.
      warnings.warn(message, FutureWarning)



```python
# Define the different metrics that we will use
def rmse(y_test, y_pred):
    """
    Compte the Root Mean Square Error
    """
    return math.sqrt(mean_squared_error(y_test, y_pred))

def mae(y_test, y_pred):
    """
    Compte the Mean Absolute Error
    """
    return mean_absolute_error(y_test, y_pred)

def asym_loss(y_test, y_pred, over_loss, under_loss):
    """
    Compte the Mean Asymetric Error
    """
    y_diff = y_pred - y_test
    over_pred_loss = (y_diff[y_diff > 0]).sum() * over_loss
    under_pred_loss = (y_diff[y_diff < 0]).sum() * under_loss
    return over_pred_loss - under_pred_loss

```


```python
df = pd.read_csv("data/weekly_newspaper.csv")
print(df.shape)
df.head()
```

    (64177, 10)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>selling_date</th>
      <th>selling_week</th>
      <th>selling_month</th>
      <th>selling_year</th>
      <th>location</th>
      <th>is_temporary</th>
      <th>is_holiday</th>
      <th>selling_store</th>
      <th>sold_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8559776</td>
      <td>20041130</td>
      <td>48</td>
      <td>11</td>
      <td>2004</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>452</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8559767</td>
      <td>20041130</td>
      <td>48</td>
      <td>11</td>
      <td>2004</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>451</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8559681</td>
      <td>20041130</td>
      <td>48</td>
      <td>11</td>
      <td>2004</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>1617</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8559671</td>
      <td>20041130</td>
      <td>48</td>
      <td>11</td>
      <td>2004</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>1616</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8559662</td>
      <td>20041130</td>
      <td>48</td>
      <td>11</td>
      <td>2004</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>1615</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictor_cols = [
    "selling_week",
    "selling_month",
    "selling_year",
    "location",
    "is_temporary",
    "is_holiday",
    "selling_store",
]
    
date_column = "selling_date"
target_column = "sold_number"
```


```python
# Load the X dataset and y target train validation test
def split_X_y_train_test(
        df: pd.DataFrame(),
        predictor_cols: list,
        target_col: str,
        col_to_split_on : str,
        value_to_split_on: int):
    """
    Split X features and y target for train and test
    """
    df_train = df[df[col_to_split_on] < value_to_split_on]
    df_test = df[df[col_to_split_on] >= value_to_split_on]
    X_train = df_train[predictor_cols]
    y_train = df_train[target_col]
    X_test = df_test[predictor_cols]
    y_test = df_test[target_col]

    return X_train, y_train, X_test, y_test

# Do the split
X_train, y_train, X_test, y_test = split_X_y_train_test(
    df, predictor_cols, target_column,
    date_column, 20040101)

# Split again to get the validation and test
X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test, test_size=0.2, random_state=RANDOM_STATE)
```


```python
OVER_LOSS = 0.5
UNDER_LOSS = 1.5
```

# Straitforward model
Here, we present the results of the training of a model without taking into account the specificity of the asymetric error evaluation.
Note that we will use a Random Forest Regressor model and which will be our "control" model.



```python
# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, criterion="mse", random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

```


```python
# Evaluation of the test dataset
y_valid_pred = model.predict(X_valid)

print("RMSE: ", rmse(y_valid, y_valid_pred))
print("MAE: ", mae(y_valid, y_valid_pred))
total_amount = asym_loss(y_valid, y_valid_pred, OVER_LOSS, UNDER_LOSS)
mean_amount = total_amount / len(y_valid)
print("Mean asymmetric money loss (compared to optimum): ", mean_amount)

```

    RMSE:  2.5710727304516565
    MAE:  1.6628447574934075
    Mean asymmetric money loss (compared to optimum):  1.3819756144402426



```python
# Evaluation of the validation dataset
y_test_pred = model.predict(X_test)

print("RMSE: ", rmse(y_test, y_test_pred))
print("MAE: ", mae(y_test, y_test_pred))
total_amount = asym_loss(y_test, y_test_pred, OVER_LOSS, UNDER_LOSS)
mean_amount = total_amount / len(y_test)
print("Mean asymmetric money loss (compared to optimum): ", mean_amount)

```

    RMSE:  2.5204673713528654
    MAE:  1.6410079780782227
    Mean asymmetric money loss (compared to optimum):  1.3924949688564525


In what follows, we will see 3 different techniques to tackle the asymetric evaluation of the error:
- Generalized increase of all predictions
- Selection by higher quantile in regression trees
- Asymmetric falsification of training data

We will explain each technique in what follows.

# Technique 1: Generalized increase of all predictions
This technique is the easiest to understand and to implement.
Its principle is straightforward: you systematically multiply the predicted value by a factor.
And then you evaluate for each factor multiplication, the gain in the performances.



```python
# First, train a model
model = RandomForestRegressor(n_estimators=10, criterion="mse", random_state=RANDOM_STATE)
model.fit(X_train, y_train)
```




    RandomForestRegressor(n_estimators=10, random_state=42)




```python
# Compute the asymetric loss (on the validation set) by varying the factor of multiplication
percentage_list = []
mean_asym_loss_list = []

for per_to_aug in range(1, 100):
    percentage_list.append(per_to_aug)
    
    rate_to_aug = 1 + (per_to_aug / 100)
    print("rate: ", rate_to_aug)
    
    y_valid_pred = model.predict(X_valid)
    y_valid_pred = y_valid_pred * rate_to_aug
    
    print("RMSE: ", rmse(y_valid, y_valid_pred))
    print("MAE: ", mae(y_valid, y_valid_pred))
    total_amount = asym_loss(y_valid, y_valid_pred, OVER_LOSS, UNDER_LOSS)
    mean_amount = total_amount / len(y_valid)
    print("Mean asymmetric money loss: ", mean_amount)
    mean_asym_loss_list.append(mean_amount)
    print()

evol_loss_df = pd.DataFrame({
        "percentage": percentage_list,
        "mean_asym_loss": mean_asym_loss_list,
    })
```

    rate:  1.01
    RMSE:  2.5991208535396724
    MAE:  1.679915714342949
    Mean asymmetric money loss:  1.3755414949827822
    
    rate:  1.02
    RMSE:  2.6286678521975726
    MAE:  1.6976866475126422
    Mean asymmetric money loss:  1.3698073518454734
    
    rate:  1.03
    RMSE:  2.659663772526285
    MAE:  1.7162380513851005
    Mean asymmetric money loss:  1.3648536794109298
    
    rate:  1.04
    RMSE:  2.692058567052367
    MAE:  1.7356437268391094
    Mean asymmetric money loss:  1.360754278557937
    
    rate:  1.05
    RMSE:  2.7258023614765037
    MAE:  1.7558421338951915
    Mean asymmetric money loss:  1.3574476093070178
    
    rate:  1.06
    RMSE:  2.760845692729181
    MAE:  1.7768554092650057
    Mean asymmetric money loss:  1.3549558083698297
    
    rate:  1.07
    RMSE:  2.797139718378616
    MAE:  1.7985265700243531
    Mean asymmetric money loss:  1.3531218928221753
    
    rate:  1.08
    RMSE:  2.8346363980307387
    MAE:  1.8209158030185144
    Mean asymmetric money loss:  1.3520060495093351
    
    rate:  1.09
    RMSE:  2.8732886478371262
    MAE:  1.844016916342991
    Mean asymmetric money loss:  1.3516020865268097
    
    rate:  1.1
    RMSE:  2.9130504695899573
    MAE:  1.8677690928174888
    Mean asymmetric money loss:  1.3518491866943056
    
    rate:  1.11
    RMSE:  2.9538770561427206
    MAE:  1.8920453374147759
    Mean asymmetric money loss:  1.3526203549845912
    
    rate:  1.12
    RMSE:  2.9957248750634697
    MAE:  1.9177055437156318
    Mean asymmetric money loss:  1.3547754849784452
    
    rate:  1.13
    RMSE:  3.038551732516865
    MAE:  1.9438118285071546
    Mean asymmetric money loss:  1.3573766934629663
    
    rate:  1.1400000000000001
    RMSE:  3.082316819395436
    MAE:  1.9705040343132285
    Mean asymmetric money loss:  1.3605638229620385
    
    rate:  1.15
    RMSE:  3.1269807416922615
    MAE:  1.997790784545328
    Mean asymmetric money loss:  1.3643454968871365
    
    rate:  1.16
    RMSE:  3.172505537038529
    MAE:  2.0258241446124585
    Mean asymmetric money loss:  1.368873780647265
    
    rate:  1.17
    RMSE:  3.2188546792307218
    MAE:  2.05452140869727
    Mean asymmetric money loss:  1.3740659684250747
    
    rate:  1.18
    RMSE:  3.265993072452916
    MAE:  2.0837826346838364
    Mean asymmetric money loss:  1.3798221181046395
    
    rate:  1.19
    RMSE:  3.3138870367673845
    MAE:  2.1136186232141445
    Mean asymmetric money loss:  1.386153030327946
    
    rate:  1.2
    RMSE:  3.3625042863078907
    MAE:  2.1439787614311645
    Mean asymmetric money loss:  1.3930080922379642
    
    rate:  1.21
    RMSE:  3.4118139014697895
    MAE:  2.1749968245041327
    Mean asymmetric money loss:  1.4005210790039302
    
    rate:  1.22
    RMSE:  3.4617862962530803
    MAE:  2.206432802697999
    Mean asymmetric money loss:  1.4084519808907952
    
    rate:  1.23
    RMSE:  3.512393181781942
    MAE:  2.238298101242628
    Mean asymmetric money loss:  1.4168122031284225
    
    rate:  1.24
    RMSE:  3.5636075268988976
    MAE:  2.270506421561442
    Mean asymmetric money loss:  1.4255154471402343
    
    rate:  1.25
    RMSE:  3.6154035166150167
    MAE:  2.3029800255310113
    Mean asymmetric money loss:  1.4344839748028015
    
    rate:  1.26
    RMSE:  3.6677565090901987
    MAE:  2.336503328992028
    Mean asymmetric money loss:  1.4445022019568166
    
    rate:  1.27
    RMSE:  3.7206429917198935
    MAE:  2.370169527182963
    Mean asymmetric money loss:  1.4546633238407494
    
    rate:  1.28
    RMSE:  3.7740405368165755
    MAE:  2.4041632360956062
    Mean asymmetric money loss:  1.4651519564463917
    
    rate:  1.29
    RMSE:  3.8279277572956723
    MAE:  2.4385738403921837
    Mean asymmetric money loss:  1.4760574844359668
    
    rate:  1.3
    RMSE:  3.882284262705933
    MAE:  2.473377789899605
    Mean asymmetric money loss:  1.4873563576363864
    
    rate:  1.31
    RMSE:  3.9370906158828975
    MAE:  2.50853690556063
    Mean asymmetric money loss:  1.4990103969904098
    
    rate:  1.32
    RMSE:  3.992328290450541
    MAE:  2.544020790343133
    Mean asymmetric money loss:  1.5109892054659106
    
    rate:  1.33
    RMSE:  4.0479796293496655
    MAE:  2.579851851961481
    Mean asymmetric money loss:  1.523315190777257
    
    rate:  1.34
    RMSE:  4.1040278045314835
    MAE:  2.6161368792027035
    Mean asymmetric money loss:  1.536095141711478
    
    rate:  1.35
    RMSE:  4.160456777920509
    MAE:  2.652804354414037
    Mean asymmetric money loss:  1.5492575406158093
    
    rate:  1.3599999999999999
    RMSE:  4.217251263721601
    MAE:  2.6897999244500066
    Mean asymmetric money loss:  1.5627480343447782
    
    rate:  1.37
    RMSE:  4.274396692121329
    MAE:  2.7270681205811713
    Mean asymmetric money loss:  1.5765111541689405
    
    rate:  1.38
    RMSE:  4.331879174413061
    MAE:  2.764653603001702
    Mean asymmetric money loss:  1.59059156028247
    
    rate:  1.3900000000000001
    RMSE:  4.389685469558039
    MAE:  2.802531246076851
    Mean asymmetric money loss:  1.6049641270506163
    
    rate:  1.4
    RMSE:  4.447802952180312
    MAE:  2.840659602776522
    Mean asymmetric money loss:  1.6195874074432859
    
    rate:  1.41
    RMSE:  4.506219581982037
    MAE:  2.879051685536225
    Mean asymmetric money loss:  1.6344744138959872
    
    rate:  1.42
    RMSE:  4.564923874556113
    MAE:  2.917640105812702
    Mean asymmetric money loss:  1.6495577578654625
    
    rate:  1.43
    RMSE:  4.623904873565843
    MAE:  2.9564333101703064
    Mean asymmetric money loss:  1.6648458859160655
    
    rate:  1.44
    RMSE:  4.683152124255528
    MAE:  2.9959186998020524
    Mean asymmetric money loss:  1.6808261992408096
    
    rate:  1.45
    RMSE:  4.742655648251486
    MAE:  3.035537510157049
    Mean asymmetric money loss:  1.696939933288804
    
    rate:  1.46
    RMSE:  4.80240591960987
    MAE:  3.0753358870580567
    Mean asymmetric money loss:  1.7132332338828105
    
    rate:  1.47
    RMSE:  4.862393842065416
    MAE:  3.1153018496293745
    Mean asymmetric money loss:  1.729694120147126
    
    rate:  1.48
    RMSE:  4.922610727433865
    MAE:  3.1555298191994026
    Mean asymmetric money loss:  1.7464170134101522
    
    rate:  1.49
    RMSE:  4.98304827512014
    MAE:  3.195977842969972
    Mean asymmetric money loss:  1.7633599608737205
    
    rate:  1.5
    RMSE:  5.04369855268415
    MAE:  3.2365584870473283
    Mean asymmetric money loss:  1.7804355286440743
    
    rate:  1.51
    RMSE:  5.104553977416451
    MAE:  3.277527197796612
    Mean asymmetric money loss:  1.7978991630863568
    
    rate:  1.52
    RMSE:  5.165607298876638
    MAE:  3.3186416501198175
    Mean asymmetric money loss:  1.8155085391025598
    
    rate:  1.53
    RMSE:  5.22685158234827
    MAE:  3.3599052274162413
    Mean asymmetric money loss:  1.8332670400919826
    
    rate:  1.54
    RMSE:  5.288280193165371
    MAE:  3.4012853257294715
    Mean asymmetric money loss:  1.8511420620982102
    
    rate:  1.55
    RMSE:  5.349886781866817
    MAE:  3.442933794342908
    Mean asymmetric money loss:  1.8692854544046456
    
    rate:  1.56
    RMSE:  5.411665270136443
    MAE:  3.4846510194636053
    Mean asymmetric money loss:  1.8874976032183408
    
    rate:  1.5699999999999998
    RMSE:  5.473609837488248
    MAE:  3.526482988860732
    Mean asymmetric money loss:  1.905824496308466
    
    rate:  1.58
    RMSE:  5.5357149086576785
    MAE:  3.5684167763440025
    Mean asymmetric money loss:  1.9242532074847345
    
    rate:  1.5899999999999999
    RMSE:  5.597975141661596
    MAE:  3.610547030267664
    Mean asymmetric money loss:  1.9428783851013942
    
    rate:  1.6
    RMSE:  5.660385416491268
    MAE:  3.6527916894324663
    Mean asymmetric money loss:  1.9616179679591947
    
    rate:  1.6099999999999999
    RMSE:  5.722940824404215
    MAE:  3.695172013080126
    Mean asymmetric money loss:  1.980493215299853
    
    rate:  1.62
    RMSE:  5.785636657782559
    MAE:  3.7376506107299443
    Mean asymmetric money loss:  1.9994667366426693
    
    rate:  1.63
    RMSE:  5.84846840052691
    MAE:  3.780219361509448
    Mean asymmetric money loss:  2.018530411115172
    
    rate:  1.6400000000000001
    RMSE:  5.911431718956571
    MAE:  3.8228685711517945
    Mean asymmetric money loss:  2.0376745444505158
    
    rate:  1.65
    RMSE:  5.974522453188152
    MAE:  3.865594885011783
    Mean asymmetric money loss:  2.0568957820035028
    
    rate:  1.6600000000000001
    RMSE:  6.037736608966352
    MAE:  3.9083456056539827
    Mean asymmetric money loss:  2.0761414263387006
    
    rate:  1.67
    RMSE:  6.101070349921819
    MAE:  3.951288431885753
    Mean asymmetric money loss:  2.095579176263469
    
    rate:  1.6800000000000002
    RMSE:  6.164519990232602
    MAE:  3.994625322368847
    Mean asymmetric money loss:  2.1154109904395604
    
    rate:  1.69
    RMSE:  6.228081987666756
    MAE:  4.038014041462458
    Mean asymmetric money loss:  2.13529463322617
    
    rate:  1.7
    RMSE:  6.29175293698508
    MAE:  4.081451046398151
    Mean asymmetric money loss:  2.155226561854862
    
    rate:  1.71
    RMSE:  6.355529563683991
    MAE:  4.124957680612662
    Mean asymmetric money loss:  2.175228119762371
    
    rate:  1.72
    RMSE:  6.419408718059755
    MAE:  4.168541048060525
    Mean asymmetric money loss:  2.195306410903232
    
    rate:  1.73
    RMSE:  6.4833873695763025
    MAE:  4.212216677460133
    Mean asymmetric money loss:  2.215476963995838
    
    rate:  1.74
    RMSE:  6.547462601519844
    MAE:  4.255967738077299
    Mean asymmetric money loss:  2.2357229483060026
    
    rate:  1.75
    RMSE:  6.611631605924519
    MAE:  4.299798223224032
    Mean asymmetric money loss:  2.256048357145733
    
    rate:  1.76
    RMSE:  6.675891678754127
    MAE:  4.34368260107474
    Mean asymmetric money loss:  2.27642765868944
    
    rate:  1.77
    RMSE:  6.740240215325912
    MAE:  4.387655192938655
    Mean asymmetric money loss:  2.296895174246354
    
    rate:  1.78
    RMSE:  6.804674705963126
    MAE:  4.431695600505694
    Mean asymmetric money loss:  2.3174305055063904
    
    rate:  1.79
    RMSE:  6.86919273186388
    MAE:  4.475772670679997
    Mean asymmetric money loss:  2.338002499373691
    
    rate:  1.8
    RMSE:  6.933791961174514
    MAE:  4.519887312880505
    Mean asymmetric money loss:  2.358612065267198
    
    rate:  1.81
    RMSE:  6.998470145256361
    MAE:  4.564028529133162
    Mean asymmetric money loss:  2.3792482052128525
    
    rate:  1.8199999999999998
    RMSE:  7.063225115135465
    MAE:  4.608228787141283
    Mean asymmetric money loss:  2.3999433869139732
    
    rate:  1.83
    RMSE:  7.128054778125362
    MAE:  4.65261094585191
    Mean asymmetric money loss:  2.420820469317597
    
    rate:  1.8399999999999999
    RMSE:  7.192957114613622
    MAE:  4.697045016052522
    Mean asymmetric money loss:  2.4417494632112082
    
    rate:  1.85
    RMSE:  7.2579301750034215
    MAE:  4.741531280585758
    Mean asymmetric money loss:  2.4627306514374427
    
    rate:  1.8599999999999999
    RMSE:  7.322972076801823
    MAE:  4.7860565379361075
    Mean asymmetric money loss:  2.4837508324807898
    
    rate:  1.87
    RMSE:  7.388081001847015
    MAE:  4.830608527204262
    Mean asymmetric money loss:  2.504797745441943
    
    rate:  1.88
    RMSE:  7.453255193667118
    MAE:  4.875213785607048
    Mean asymmetric money loss:  2.525897927537727
    
    rate:  1.8900000000000001
    RMSE:  7.518492954963684
    MAE:  4.919883434779957
    Mean asymmetric money loss:  2.5470625004036336
    
    rate:  1.9
    RMSE:  7.5837926452132605
    MAE:  4.964609532952861
    Mean asymmetric money loss:  2.568283522269536
    
    rate:  1.9100000000000001
    RMSE:  7.649152678380952
    MAE:  5.00937734712545
    Mean asymmetric money loss:  2.589546260135123
    
    rate:  1.92
    RMSE:  7.714571520740071
    MAE:  5.054183159561516
    Mean asymmetric money loss:  2.6108469962641876
    
    rate:  1.9300000000000002
    RMSE:  7.780047688792442
    MAE:  5.0990401046836675
    Mean asymmetric money loss:  2.632198865079337
    
    rate:  1.94
    RMSE:  7.845579747284135
    MAE:  5.143940370773008
    Mean asymmetric money loss:  2.653594054861675
    
    rate:  1.95
    RMSE:  7.911166307311768
    MAE:  5.188877524804043
    Mean asymmetric money loss:  2.67502613258571
    
    rate:  1.96
    RMSE:  7.9768060245147225
    MAE:  5.233836687938665
    Mean asymmetric money loss:  2.69648021941333
    
    rate:  1.97
    RMSE:  8.042497597348936
    MAE:  5.27881391090437
    Mean asymmetric money loss:  2.717952366072034
    
    rate:  1.98
    RMSE:  8.108239765438118
    MAE:  5.323803257952483
    Mean asymmetric money loss:  2.7394366368131435
    
    rate:  1.99
    RMSE:  8.174031307998524
    MAE:  5.368798630206477
    Mean asymmetric money loss:  2.7609269327601362
    



```python
# Plot the mean asymetric loss according the factor
plt.plot(evol_loss_df['percentage'], evol_loss_df['mean_asym_loss'])
plt.grid(True)
plt.xlabel('Percentage')
plt.ylabel('Mean asymmetric money loss')
plt.show()
```


    
![png](output_15_0.png)
    



```python
# Determine the best multiplication factor
index_min = evol_loss_df.idxmin(axis = 0)["mean_asym_loss"]
best_percentage = evol_loss_df.iloc[index_min]['percentage']
best_percentage
```




    9.0




```python
# Compute the mean asymetric loss on the test dataset according the the best multiplication factor
rate_to_aug = 1 + (best_percentage / 100)

y_valid_pred = model.predict(X_valid)
y_valid_pred = y_valid_pred * rate_to_aug

print("RMSE: ", rmse(y_valid, y_valid_pred))
print("MAE: ", mae(y_valid, y_valid_pred))
total_amount = asym_loss(y_valid, y_valid_pred, OVER_LOSS, UNDER_LOSS)
mean_amount = total_amount / len(y_valid)
print("Mean asymmetric money loss (compared to optimum): ", mean_amount)

```

    RMSE:  2.8732886478371262
    MAE:  1.844016916342991
    Mean asymmetric money loss (compared to optimum):  1.3516020865268097


We notice a reducing of the loss from 1.392€ to 1.351€.

# Technique 2: Selection by higher quantile in regression trees
This technique is based on the the selection of a specific quantile instead of the median or average in the classical decision trees approach.
We base on a library which implements this approach.


```python
# Train a the Random Forest "Quantile" Regressor that extends the standart scikit implementation
model = RandomForestQuantileRegressor(n_estimators=10, criterion="mse", random_state=RANDOM_STATE)
model.fit(X_train, y_train)
```

    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)
    /Users/nabil.belaid/anaconda3/lib/python3.6/site-packages/sklearn/tree/_classes.py:327: FutureWarning: The parameter 'presort' is deprecated and has no effect. It will be removed in v0.24. You can suppress this warning by not passing any value to the 'presort' parameter.
      FutureWarning)





    RandomForestQuantileRegressor(random_state=42)




```python
# Compute the mean asymetric loss (on the validation set) by varying the quantile
# Note that this specific implementation (at this current version) is slow
percentage_list = []
mean_asym_loss_list = []

#for i in range(1, 100):
for i in [10,20,30,40,50,60,70,80,90]:
    percentage_list.append(i)
    print("Quantile:", i)
    y_valid_pred = model.predict(X_valid, quantile=i)

    print("RMSE:", rmse(y_valid, y_valid_pred))
    print("MAE:", mae(y_valid, y_valid_pred))
    total_amount = asym_loss(y_valid, y_valid_pred, OVER_LOSS, UNDER_LOSS)
    mean_amount = total_amount / len(y_valid)
    print("Mean asymmetric money loss (compared to optimum):", mean_amount)
    mean_asym_loss_list.append(mean_amount)
    print()
    

evol_loss_df = pd.DataFrame({
        "percentage": percentage_list,
        "mean_asym_loss": mean_asym_loss_list,
    })
```

    Quantile: 10
    RMSE: 2.648219751032231
    MAE: 1.6774242579125813
    Mean asymmetric money loss (compared to optimum): 2.029805467573433
    
    Quantile: 20
    RMSE: 2.54581791239489
    MAE: 1.6207645853694155
    Mean asymmetric money loss (compared to optimum): 1.8018224421383269
    
    Quantile: 30
    RMSE: 2.524632615895418
    MAE: 1.6132646293573543
    Mean asymmetric money loss (compared to optimum): 1.6398686586166447
    
    Quantile: 40
    RMSE: 2.5534726472838116
    MAE: 1.638215909075281
    Mean asymmetric money loss (compared to optimum): 1.516808071631148
    
    Quantile: 50
    RMSE: 2.639319434449881
    MAE: 1.6962092990971038
    Mean asymmetric money loss (compared to optimum): 1.427544909612238
    
    Quantile: 60
    RMSE: 2.7885374066281847
    MAE: 1.7834933756640887
    Mean asymmetric money loss (compared to optimum): 1.3647182932708428
    
    Quantile: 70
    RMSE: 2.995019384446774
    MAE: 1.9062070757178657
    Mean asymmetric money loss (compared to optimum): 1.3302375629426126
    
    Quantile: 80
    RMSE: 3.2451853046602555
    MAE: 2.0673566385995157
    Mean asymmetric money loss (compared to optimum): 1.3260045903432096
    
    Quantile: 90
    RMSE: 3.577361648551947
    MAE: 2.2959163765355126
    Mean asymmetric money loss (compared to optimum): 1.3680665364490538
    



```python
# Plot the mean asymetric loss according the factor
plt.plot(evol_loss_df['percentage'], evol_loss_df['mean_asym_loss'])
plt.grid(True)
plt.xlabel('Percentage')
plt.ylabel('Mean asymmetric money loss')
plt.show()
```


    
![png](output_22_0.png)
    



```python
# Determine the quantile that gives the best outcome on the validation dataset
index_min = evol_loss_df.idxmin(axis = 0)["mean_asym_loss"]
best_percentage = evol_loss_df.iloc[index_min]['percentage']
best_percentage
```




    80.0




```python
# Compute the mean asymetric loss on the test dataset on the best retained quantile
y_test_pred = model.predict(X_test, quantile=best_percentage)

print("RMSE: ", rmse(y_test, y_test_pred))
print("MAE: ", mae(y_test, y_test_pred))
total_amount = asym_loss(y_test, y_test_pred, OVER_LOSS, UNDER_LOSS)
mean_amount = total_amount / len(y_test)
print("Mean asymmetric money loss (compared to optimum): ", mean_amount)

```

    RMSE:  3.1940852239410877
    MAE:  2.032219712850274
    Mean asymmetric money loss (compared to optimum):  1.327702358861495


We notice a reducing of the loss from 1.392€ to 1.327€.

# Technique 3: Asymmetric falsification of training data
This approach is original.
It consists in training a first model.
Then, we "falsify" the training data by majorating the instances that the model under-estimated.
We then evaluate the new performance on new data.


```python
# First, train a model once
model = RandomForestRegressor(n_estimators=10, criterion="mse", random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
```


```python
# Evaluate the model after the first iteration of training
y_valid_pred = model.predict(X_valid)

print("RMSE: ", rmse(y_valid, y_valid_pred))
print("MAE: ", mae(y_valid, y_valid_pred))
total_amount = asym_loss(y_valid, y_valid_pred, OVER_LOSS, UNDER_LOSS)
mean_amount = total_amount / len(y_valid)
print("Mean asymmetric money loss (compared to optimum): ", mean_amount)

```

    RMSE:  2.5710727304516565
    MAE:  1.6628447574934075
    Mean asymmetric money loss (compared to optimum):  1.3819756144402426



```python
# Save the original train target
y_train_it_0 = y_train
```


```python
# Apply different majorations to the underestimated instances and compute the mean asymetric loss (on the validation set)
percentage_list = []
mean_asym_loss_list = []

for per_to_aug in range(1, 100):
    percentage_list.append(per_to_aug)
    
    rate_to_aug = per_to_aug / 100
    print("rate: ", rate_to_aug)
    
    y_train = y_train_it_0
    to_be_aug = (y_train_pred - y_train)<0
    y_train = y_train * (1 + (rate_to_aug * to_be_aug))
    
    model = RandomForestRegressor(n_estimators=10, criterion="mse", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    
    y_valid_pred = model.predict(X_valid)
    print("RMSE: ", rmse(y_valid, y_valid_pred))
    print("MAE: ", mae(y_valid, y_valid_pred))
    total_amount = asym_loss(y_valid, y_valid_pred, OVER_LOSS, UNDER_LOSS)
    mean_amount = total_amount / len(y_valid)
    print("Asymmetric money loss:", mean_amount)
    mean_asym_loss_list.append(mean_amount)
    print()

evol_loss_df = pd.DataFrame({
        "percentage": percentage_list,
        "mean_asym_loss": mean_asym_loss_list,
    })
```

    rate:  0.01
    RMSE:  2.5870904720413557
    MAE:  1.673015967291056
    Asymmetric money loss: 1.3791226102061174
    
    rate:  0.02
    RMSE:  2.606494660382465
    MAE:  1.6837453970030811
    Asymmetric money loss: 1.3774102778970883
    
    rate:  0.03
    RMSE:  2.6218258881483867
    MAE:  1.6932820392962473
    Asymmetric money loss: 1.3742183123250136
    
    rate:  0.04
    RMSE:  2.642492131237598
    MAE:  1.7035349697543753
    Asymmetric money loss: 1.3716849054207036
    
    rate:  0.05
    RMSE:  2.654724827171211
    MAE:  1.7112059230203323
    Asymmetric money loss: 1.3721986033103721
    
    rate:  0.06
    RMSE:  2.672568458878923
    MAE:  1.7229453048464416
    Asymmetric money loss: 1.3715189680483784
    
    rate:  0.07
    RMSE:  2.6853591094461535
    MAE:  1.728821544262694
    Asymmetric money loss: 1.370322909198374
    
    rate:  0.08
    RMSE:  2.6996301589694296
    MAE:  1.738837859755158
    Asymmetric money loss: 1.3698161410429406
    
    rate:  0.09
    RMSE:  2.7106204318928024
    MAE:  1.7446898332467125
    Asymmetric money loss: 1.3695575658425647
    
    rate:  0.1
    RMSE:  2.7310915189592557
    MAE:  1.7565619691895447
    Asymmetric money loss: 1.3708684227768706
    
    rate:  0.11
    RMSE:  2.7408869032086525
    MAE:  1.760288352470273
    Asymmetric money loss: 1.3680303114133565
    
    rate:  0.12
    RMSE:  2.7663157679156534
    MAE:  1.7754009567102986
    Asymmetric money loss: 1.373038289993174
    
    rate:  0.13
    RMSE:  2.754764653294931
    MAE:  1.772607712497595
    Asymmetric money loss: 1.3677306345011975
    
    rate:  0.14
    RMSE:  2.7877201472302344
    MAE:  1.7898479485836345
    Asymmetric money loss: 1.3753548954982888
    
    rate:  0.15
    RMSE:  2.793535822620104
    MAE:  1.796517983785625
    Asymmetric money loss: 1.3757676050675103
    
    rate:  0.16
    RMSE:  2.8223547695130886
    MAE:  1.8035774899121455
    Asymmetric money loss: 1.3746521962888587
    
    rate:  0.17
    RMSE:  2.822003330193142
    MAE:  1.8100978542828234
    Asymmetric money loss: 1.3752287892811836
    
    rate:  0.18
    RMSE:  2.8595668734528283
    MAE:  1.8228718373478023
    Asymmetric money loss: 1.3795067738091533
    
    rate:  0.19
    RMSE:  2.8467859145013326
    MAE:  1.8243378572180313
    Asymmetric money loss: 1.3778201106281174
    
    rate:  0.2
    RMSE:  2.879896168397999
    MAE:  1.8362481001449151
    Asymmetric money loss: 1.3785125169042114
    
    rate:  0.21
    RMSE:  2.8755434881406834
    MAE:  1.8389383988134107
    Asymmetric money loss: 1.377598609681664
    
    rate:  0.22
    RMSE:  2.9144402237851286
    MAE:  1.8568175360041335
    Asymmetric money loss: 1.3852369941422276
    
    rate:  0.23
    RMSE:  2.8908881019282355
    MAE:  1.851120733022864
    Asymmetric money loss: 1.3786539717592465
    
    rate:  0.24
    RMSE:  2.946843491367442
    MAE:  1.8735641824865483
    Asymmetric money loss: 1.3878965600179372
    
    rate:  0.25
    RMSE:  2.92487199722413
    MAE:  1.8681538245774276
    Asymmetric money loss: 1.3847660071974095
    
    rate:  0.26
    RMSE:  2.9692985429319214
    MAE:  1.8887049892552261
    Asymmetric money loss: 1.3927767111053933
    
    rate:  0.27
    RMSE:  2.9576484372999388
    MAE:  1.8868824270153242
    Asymmetric money loss: 1.3889161071932294
    
    rate:  0.28
    RMSE:  2.9999725622104596
    MAE:  1.9056525462841942
    Asymmetric money loss: 1.3992321473846754
    
    rate:  0.29
    RMSE:  2.994183132184814
    MAE:  1.9029301445410434
    Asymmetric money loss: 1.3937628991518083
    
    rate:  0.3
    RMSE:  3.030087236234648
    MAE:  1.9209237128738077
    Asymmetric money loss: 1.4011107756323538
    
    rate:  0.31
    RMSE:  3.025391707927127
    MAE:  1.9226318654646326
    Asymmetric money loss: 1.3988356558762758
    
    rate:  0.32
    RMSE:  3.0672088908226947
    MAE:  1.9359368416583547
    Asymmetric money loss: 1.4031145494359265
    
    rate:  0.33
    RMSE:  3.0450864637158848
    MAE:  1.9354020039541588
    Asymmetric money loss: 1.4011053080504536
    
    rate:  0.34
    RMSE:  3.1212038801610196
    MAE:  1.9649270173402782
    Asymmetric money loss: 1.4169935668036544
    
    rate:  0.35
    RMSE:  3.0897136662254563
    MAE:  1.953013016399235
    Asymmetric money loss: 1.4065472268465697
    
    rate:  0.36
    RMSE:  3.152678573405531
    MAE:  1.9817606183561307
    Asymmetric money loss: 1.4175118740515673
    
    rate:  0.37
    RMSE:  3.114995824235963
    MAE:  1.9704966332329759
    Asymmetric money loss: 1.4167215591736517
    
    rate:  0.38
    RMSE:  3.179598083084014
    MAE:  1.9963150479741885
    Asymmetric money loss: 1.4233264014107907
    
    rate:  0.39
    RMSE:  3.1499825456648853
    MAE:  1.9911583078680997
    Asymmetric money loss: 1.4253572867073814
    
    rate:  0.4
    RMSE:  3.198705212585427
    MAE:  2.013948061942793
    Asymmetric money loss: 1.427514307899194
    
    rate:  0.41
    RMSE:  3.2016189868243643
    MAE:  2.0065583476164166
    Asymmetric money loss: 1.428212906723323
    
    rate:  0.42
    RMSE:  3.2421953713204688
    MAE:  2.033023359350414
    Asymmetric money loss: 1.4334955392387845
    
    rate:  0.43
    RMSE:  3.237026841691846
    MAE:  2.023023767144398
    Asymmetric money loss: 1.4315501881531392
    
    rate:  0.44
    RMSE:  3.26242932184272
    MAE:  2.0458263719533547
    Asymmetric money loss: 1.4403155921864244
    
    rate:  0.45
    RMSE:  3.293132840803421
    MAE:  2.051268402155242
    Asymmetric money loss: 1.4436786915007225
    
    rate:  0.46
    RMSE:  3.311671067797607
    MAE:  2.0724151473293198
    Asymmetric money loss: 1.4515261530856072
    
    rate:  0.47
    RMSE:  3.317998013324506
    MAE:  2.068011613838052
    Asymmetric money loss: 1.4474034088445997
    
    rate:  0.48
    RMSE:  3.353343910231896
    MAE:  2.0907154968889876
    Asymmetric money loss: 1.4587334130890903
    
    rate:  0.49
    RMSE:  3.372930115060701
    MAE:  2.087504744050908
    Asymmetric money loss: 1.454010135971763
    
    rate:  0.5
    RMSE:  3.390990289309924
    MAE:  2.110068096222193
    Asymmetric money loss: 1.4646410686963909
    
    rate:  0.51
    RMSE:  3.415499615717422
    MAE:  2.106239772748629
    Asymmetric money loss: 1.4640623248684994
    
    rate:  0.52
    RMSE:  3.429529945372337
    MAE:  2.133381643350729
    Asymmetric money loss: 1.4729865357594758
    
    rate:  0.53
    RMSE:  3.5170208832106793
    MAE:  2.1380760807219725
    Asymmetric money loss: 1.4766246779956347
    
    rate:  0.54
    RMSE:  3.4841589331312273
    MAE:  2.1563718560559075
    Asymmetric money loss: 1.4818389128278604
    
    rate:  0.55
    RMSE:  3.5373755813678294
    MAE:  2.160002330397784
    Asymmetric money loss: 1.4820409679602198
    
    rate:  0.56
    RMSE:  3.5099763030152484
    MAE:  2.1712702693807713
    Asymmetric money loss: 1.488935380019994
    
    rate:  0.57
    RMSE:  3.59802505639007
    MAE:  2.1837519237057834
    Asymmetric money loss: 1.4895143265161794
    
    rate:  0.58
    RMSE:  3.5492488948428607
    MAE:  2.1991915956527492
    Asymmetric money loss: 1.5000880958104437
    
    rate:  0.59
    RMSE:  3.6837219234778362
    MAE:  2.2119092533706195
    Asymmetric money loss: 1.5041646784079532
    
    rate:  0.6
    RMSE:  3.584567387338056
    MAE:  2.2149774999581417
    Asymmetric money loss: 1.506314604626261
    
    rate:  0.61
    RMSE:  3.6851072553891573
    MAE:  2.228469955910377
    Asymmetric money loss: 1.5083670126100035
    
    rate:  0.62
    RMSE:  3.62060073746337
    MAE:  2.2332525809098507
    Asymmetric money loss: 1.5114723927507316
    
    rate:  0.63
    RMSE:  3.7444845764414003
    MAE:  2.2519988023822677
    Asymmetric money loss: 1.521652344007432
    
    rate:  0.64
    RMSE:  3.6811445840046115
    MAE:  2.2499814516567183
    Asymmetric money loss: 1.5168285539591309
    
    rate:  0.65
    RMSE:  3.754113039991949
    MAE:  2.2701231483674063
    Asymmetric money loss: 1.5242880320121026
    
    rate:  0.66
    RMSE:  3.699217889704551
    MAE:  2.2798250337457775
    Asymmetric money loss: 1.5306742271331428
    
    rate:  0.67
    RMSE:  3.8038340528202035
    MAE:  2.2904207237200667
    Asymmetric money loss: 1.5350501243946633
    
    rate:  0.68
    RMSE:  3.7777686656769935
    MAE:  2.307736772372872
    Asymmetric money loss: 1.5421522364158937
    
    rate:  0.69
    RMSE:  3.89377840952664
    MAE:  2.3185820144973293
    Asymmetric money loss: 1.544941423035595
    
    rate:  0.7
    RMSE:  3.799863276007917
    MAE:  2.3200432378485307
    Asymmetric money loss: 1.5484260054709673
    
    rate:  0.71
    RMSE:  3.9323475954476224
    MAE:  2.339534457186458
    Asymmetric money loss: 1.5580426925998023
    
    rate:  0.72
    RMSE:  3.8671407271377856
    MAE:  2.3413240438107947
    Asymmetric money loss: 1.5514059596876495
    
    rate:  0.73
    RMSE:  3.9722248759120133
    MAE:  2.3620942684430313
    Asymmetric money loss: 1.566915583118546
    
    rate:  0.74
    RMSE:  3.877486706990948
    MAE:  2.366319920777866
    Asymmetric money loss: 1.5700762183703458
    
    rate:  0.75
    RMSE:  4.008909048509568
    MAE:  2.3807976202862835
    Asymmetric money loss: 1.5726918432560895
    
    rate:  0.76
    RMSE:  4.000630691610003
    MAE:  2.392181072858873
    Asymmetric money loss: 1.5798562537836216
    
    rate:  0.77
    RMSE:  3.963956888401946
    MAE:  2.394203069222724
    Asymmetric money loss: 1.5743574315200766
    
    rate:  0.78
    RMSE:  4.005743299553302
    MAE:  2.412218023195089
    Asymmetric money loss: 1.5867111870227373
    
    rate:  0.79
    RMSE:  4.083327980120479
    MAE:  2.428627445311681
    Asymmetric money loss: 1.5909326627445706
    
    rate:  0.8
    RMSE:  4.0634041994626475
    MAE:  2.4287538970378777
    Asymmetric money loss: 1.5931666500514774
    
    rate:  0.81
    RMSE:  4.139169942233679
    MAE:  2.439065117165703
    Asymmetric money loss: 1.5959835402942406
    
    rate:  0.82
    RMSE:  4.053772779532673
    MAE:  2.4447812876689263
    Asymmetric money loss: 1.6015357662982335
    
    rate:  0.83
    RMSE:  4.201314130187783
    MAE:  2.4711644142278644
    Asymmetric money loss: 1.609234332975459
    
    rate:  0.84
    RMSE:  4.159953499550572
    MAE:  2.4766867728113877
    Asymmetric money loss: 1.6141072276643729
    
    rate:  0.85
    RMSE:  4.213277013538828
    MAE:  2.4924918238745497
    Asymmetric money loss: 1.6203376123070061
    
    rate:  0.86
    RMSE:  4.216073650870541
    MAE:  2.5013596487978917
    Asymmetric money loss: 1.6232425997512192
    
    rate:  0.87
    RMSE:  4.274112915202994
    MAE:  2.5218697807888435
    Asymmetric money loss: 1.6317187268618893
    
    rate:  0.88
    RMSE:  4.220184176394705
    MAE:  2.524881033768329
    Asymmetric money loss: 1.6367902390645286
    
    rate:  0.89
    RMSE:  4.344596471469305
    MAE:  2.5355308186080325
    Asymmetric money loss: 1.641423527326135
    
    rate:  0.9
    RMSE:  4.295377535856151
    MAE:  2.55697629497961
    Asymmetric money loss: 1.6464723739657439
    
    rate:  0.91
    RMSE:  4.415168301025458
    MAE:  2.577483713534998
    Asymmetric money loss: 1.6631891319248324
    
    rate:  0.92
    RMSE:  4.359597313501919
    MAE:  2.587064131994596
    Asymmetric money loss: 1.660153081560961
    
    rate:  0.93
    RMSE:  4.450551291928033
    MAE:  2.596248294999657
    Asymmetric money loss: 1.6717325204561886
    
    rate:  0.94
    RMSE:  4.383965117763274
    MAE:  2.600494765073465
    Asymmetric money loss: 1.6658991485135397
    
    rate:  0.95
    RMSE:  4.488687339705972
    MAE:  2.61470138600579
    Asymmetric money loss: 1.6743369442824214
    
    rate:  0.96
    RMSE:  4.44695764374281
    MAE:  2.6285588012723107
    Asymmetric money loss: 1.6773828873486167
    
    rate:  0.97
    RMSE:  4.610565068335496
    MAE:  2.648156032646413
    Asymmetric money loss: 1.6905168060135767
    
    rate:  0.98
    RMSE:  4.477168008839081
    MAE:  2.6457387523339575
    Asymmetric money loss: 1.6860502683573784
    
    rate:  0.99
    RMSE:  4.714903764423769
    MAE:  2.6861660173074746
    Asymmetric money loss: 1.7081660681340083
    



```python
# Plot the mean asymetric loss according the different majorations
plt.plot(evol_loss_df['percentage'], evol_loss_df['mean_asym_loss'])
plt.grid(True)
plt.xlabel('Percentage')
plt.ylabel('Mean asymmetric money loss')
plt.show()
```


    
![png](output_31_0.png)
    



```python
# Determine the best rate, that gives the best mean asymetric loss
index_min = evol_loss_df.idxmin(axis = 0)["mean_asym_loss"]
best_percentage = evol_loss_df.iloc[index_min]['percentage']

rate_to_aug = best_percentage / 100
print("rate: ", rate_to_aug)

y_train = y_train_it_0
to_be_aug = (y_train_pred - y_train)<0
y_train = y_train * (1 + (rate_to_aug * to_be_aug))

model = RandomForestRegressor(n_estimators=10, criterion="mse", random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

```

    rate:  0.13



```python
# Compute the mean asymetric loss on the test dataset on the best rate
y_test_pred = model.predict(X_test)
print("RMSE: ", rmse(y_test, y_test_pred))
print("MAE: ", mae(y_test, y_test_pred))
total_amount = asym_loss(y_test, y_test_pred, OVER_LOSS, UNDER_LOSS)
mean_amount = total_amount / len(y_test)
print("Asymmetric money loss:", mean_amount)
```

    RMSE:  2.6556963254081474
    MAE:  1.7179912834114055
    Asymmetric money loss: 1.3778882222296251


We notice a reducing of the loss from 1.392€ to 1.377€.

# Summary and outlooks
As a conclusion, we can say that there are different strategies to tackle the asymetric evaluation of the error.
Note however, note that some techniques are greedy and the gain does not justify their utilization.

Here is a summary about the performances of the different techniques:

|   | Mean Asymetric Loss  | Time complexity  | Implementation complexity  |   
|---|---|---|---|
|Generalized increase of all predictions | (+) |  (++) |  (++) |   
|Selection by higher quantile in regression trees | (++)  |  (-) | (-)  |   
|Asymmetric falsification of training data |  (+) | (--)  | (+)  |   


As outlooks, we site the following:
- Implement the asymmetric cost function in scikit-learn
- Use blending of techniques
- Run a second iteration of the falsification technique
- Use a "Mirror" falsification (on the other side)


```python

```
