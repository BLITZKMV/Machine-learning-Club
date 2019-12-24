# Ensure that train.csv and test.csv are in the same directory.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def y_x(slope, intercept, x):
    return slope*x + intercept

def plot_line(slope, intercept, data_X, data_Y, **kwargs):
    title = kwargs.get('title','')
    max_x = np.max(data_X)+10
    min_x = np.min(data_X)-10
    xp = np.linspace(min_x, max_x)
    yp = slope*xp + intercept
    plt.plot(xp,yp,"r-",label="Regression Estimate")
    plt.axis([np.min(data_X-10), np.max(data_X+10),
              np.min(data_Y-10), np.max(data_Y+10)])
    plt.scatter(data_X,data_Y)
    plt.hlines(y=0, xmin=np.min(data_X)-10, xmax=np.max(data_X)+10)
    plt.vlines(x=0, ymin=np.min(data_Y)-10, ymax=np.max(data_Y)+10)
    plt.title(title)
    plt.show()

def regress(X, Y):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    covariance_xy = 0
    variance_x = 0
    for (x,y) in zip(X,Y):
        covariance_xy += (x-x_bar)*(y-y_bar)
        variance_x += (x-x_bar)**2
    slope = covariance_xy / variance_x
    intercept = y_bar - (x_bar*slope)
    return slope, intercept

def test_data(slope, intercept, X, Y):
    relative_error = 0
    for (x,y) in zip(X,Y):
        relative_error += abs((y_x(slope, intercept, x)-y)/y)
    return 100.0 - (relative_error/len(Y))*100

def main():
    show_graphs = input("Display graphs [Y/n] ? ")
    show_graphs = len(show_graphs)>0 and show_graphs.lower()[0]=='y'
    print("> Loading training data from 'train.csv' ... ", end='')
    data = pd.read_csv("train.csv")
    X = list(data['x'])
    Y = list(data['y'])
    train_X = np.array(X).astype('float32')
    train_Y = np.array(Y).astype('float32')
    slope = np.random.rand()
    intercept = np.random.rand()
    if show_graphs:
        plot_line(slope, intercept, train_X, train_Y, title = 'Train Data')
    print("done.")
    print("> Calculating regression based on train data ... ", end='')
    slope, intercept = regress(train_X, train_Y)
    if show_graphs:
        plot_line(slope, intercept, train_X, train_Y, title = 'Train Data')
    print("done.\n  Slope = ", slope, ", Intercept = ", intercept)
    print("> Testing results using 'test.csv' ... ", end='')
    data = pd.read_csv("test.csv")
    X = list(data['x'])
    Y = list(data['y'])
    test_X = np.array(X).astype('float32')
    test_Y = np.array(Y).astype('float32')
    accuracy = test_data(slope, intercept, test_X, test_Y)
    if show_graphs:
        plot_line(slope, intercept, test_X, test_Y, title = 'Test Data')
    print("done.\n  Train Accuracy: ", accuracy)
    
if __name__ == "__main__":
    main()
