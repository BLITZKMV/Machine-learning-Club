import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv

# Dataset file name.
data_file = 'kc_house_data.csv'

# Training dataset share value.
train_share = 0.9 # 70% entries for training, 30% for testing.

# Relevant Properties.
properties = [ 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
               'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
               'yr_renovated', 'sqft_living15', 'sqft_lot15']

# Applies some formatting to the given data.
def normalize(data):
    pd.set_option('mode.chained_assignment',None)
    data['yr_renovated'] = data[['yr_built', 'yr_renovated']].max(axis=1)
    return data

# Returns predicted value of y from values of independent variables.
def y_x(K, Xi):
    Xi = np.array(Xi).astype('float32')
    return np.sum([k*x for (k,x) in zip(K[1:], Xi)])+K[0]

# Class to perform linear regression.
class LinearRegressionModel:
    def __init__(self, X, Y):
        """
        The regression model forms an equation of type Y = KX + E, where Y, K and E are vectors, and X is a matrix of independent variables.
        The mean squared error (MSE) is sum(e**2 for e in E)/len(E). Now, rows(E) = rows(Y) and sum(...) = E*np.transpose(E)
        Now, E = Y - KX. Thus, mean squared error is ((Y-KX)*transpose(Y-KX))/rows(Y)
        To find K which minimizes MSE, the product of transpose(X) and (Y-KX) should be zero. Thus, K may be estimated as:
           Xt(Y-KX) = 0
        => XtY - KXtX = 0
        => KXtX = XtY
        => K = (XtX)^-1 . XtY
        """
        self.X = np.array(X).astype('float32')
        self.X = np.insert(self.X, 0, values=1, axis=1)
        self.Y = np.array(Y).astype('float32')
        self.XT = np.transpose(self.X)
        self.K = inv(self.XT.dot(self.X)).dot(self.XT.dot(self.Y))
    def parameters(self):
        return np.squeeze(np.asarray(self.K))
    def predict(self, Xi):
        return y_x(self.parameters(), Xi)
    def score(self, test_X, test_Y):
        relative_error = 0
        for (Xi,y) in zip(test_X, test_Y):
            relative_error += abs((y_x(self.K, Xi)-y)/y)[0]
        return 100.0 - (relative_error/test_Y.shape[0])*100

def main():
    print("> Loading training and test data from '",data_file,"' ... ", end='')
    data = pd.read_csv(data_file)
    rows = data.shape[0]
    train_data = normalize(data.head(int(train_share*rows)))
    test_data  = normalize(data.tail(rows-int(train_share*rows)))
    train_X = train_data[properties].to_numpy()
    train_Y = train_data[['price']].to_numpy()
    test_X = test_data[properties].to_numpy()
    test_Y = test_data[['price']].to_numpy()
    print("done.\n  Dataset Sizes: Total (",rows,"), Train (",train_data.shape[0],"), Test (",test_data.shape[0],")")
    print("> Calculating regression based on train data ... ", end='')
    model = LinearRegressionModel(train_X, train_Y)
    K = model.parameters()
    print("done.")
    for (prop, rate) in zip(properties, K[1:]):
        print("  | ",prop," | ",rate," |",sep='')
    print("  | Intercept | ",K[0]," |",sep='')
    print("> Testing results using test data ... ", end='')
    accuracy = model.score(test_X, test_Y)
    print("done.\n  Train Accuracy: ", accuracy, "\n")
    dummy = input("Press ENTER or type a key to continue ... ")
    
if __name__ == "__main__":
    main()
