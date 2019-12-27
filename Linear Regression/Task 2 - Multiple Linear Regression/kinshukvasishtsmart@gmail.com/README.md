# Task 2 - Multiple Linear Regression 
###### @BLITZKMV/Machine-learning-Club

#### Theory
The regression model forms an equation of type Y = KX + E, where Y, K and E are vectors, and X is a matrix of independent variables.
The mean squared error (MSE) is ```sum(e**2 for e in E)/len(E)```. Now, ```rows(E) = rows(Y)``` and ```sum(...) = E*transpose(E)```
Now, ```E = Y - KX```. Thus, mean squared error is ```((Y-KX)*transpose(Y-KX))/rows(Y)```
To find K which minimizes MSE, the product of ```transpose(X)``` and ```(Y-KX)``` should be zero. Thus, K may be estimated as:
- ```Xt(Y-KX) = 0```, where ```Xt = transpose(X)```
- ```XtY - KXtX = 0```
- ```KXtX = XtY```
- ```K = (XtX)^-1 . XtY```

#### Calculation Results
Train : Test Ratios:
- 60:40

  | Independent Variable | Regression Coefficient |
  | --- | --- |
  | bedrooms | -43321.25 |
  | bathrooms | 45251.5 |
  | sqft_living | -440.90625 |
  | sqft_lot | 0.049171448 |
  | floors | 10740.0 |
  | waterfront | 672898.5 |
  | view | 38718.5 |
  | condition | 18319.25 |
  | grade | 109694.0 |
  | sqft_above | 611.40625 |
  | sqft_basement | 576.78125 |
  | yr_built | -4049.9375 |
  | yr_renovated | 211.6875 |
  | sqft_living15 | 36.45117 |
  | sqft_lot15 | -0.6395397 |
  | Intercept | 6783824.0 |

- 70:30

  | Independent Variable | Regression Coefficient |
  | --- | --- |
  | bedrooms | -40557.5 |
  | bathrooms | 45996.0 |
  | sqft_living | -1174.8125 |
  | sqft_lot | 0.05042839 |
  | floors | 11638.5 |
  | waterfront | 637957.5 |
  | view | 38579.25 |
  | condition | 18805.5 |
  | grade | 110761.0 |
  | sqft_above | 1395.5938 |
  | sqft_basement | 1384.0312 |
  | yr_built | -4049.1875 |
  | yr_renovated | 198.21875 |
  | sqft_living15 | 40.47705 |
  | sqft_lot15 | -0.6170778 |
  | Intercept | 6790144.0 |

- 80:20

  | Independent Variable | Regression Coefficient |
  | --- | --- |
  | bedrooms | -36633.5 |
  | bathrooms | 46378.0 |
  | sqft_living | -1247.375 |
  | sqft_lot | 0.028618813 |
  | floors | 11433.0 |
  | waterfront | 595571.5 |
  | view | 42149.25 |
  | condition | 20808.0 |
  | grade | 111746.5 |
  | sqft_above | 1304.875 |
  | sqft_basement | 1341.5938 |
  | yr_built | -4024.0 |
  | yr_renovated | 205.8125 |
  | sqft_living15 | 44.924316 |
  | sqft_lot15 | -0.60082626 |
  | Intercept | 6706304.0 |

- 90:10

  | Independent Variable | Regression Coefficient |
  | --- | --- |
  | bedrooms | -36961.0 |
  | bathrooms | 45808.0 |
  | sqft_living | -1726.5 |
  | sqft_lot | 0.01017189 |
  | floors | 13471.0 |
  | waterfront | 567113.0 |
  | view | 43103.5 |
  | condition | 20386.0 |
  | grade | 112169.25 |
  | sqft_above | 1847.5 |
  | sqft_basement | 1897.375 |
  | yr_built | -3934.25 |
  | yr_renovated | 147.875 |
  | sqft_living15 | 44.59912 |
  | sqft_lot15 | -0.59031963 |
  | Intercept | 6644864.0 |

#### Results

| Dataset Size (Train : Test) | Accuracy |
| --- | --- |
| 60 : 40 | 71.60942224459171 |
| 70 : 30 | 62.871967776811715 |
| 80 : 20 | 52.62981099417116 |
| 90 : 10 | 69.46402934307604 | 
