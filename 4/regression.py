import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

data = pd.read_csv("input.txt")
print("Type 1 for linear regression.\nType 2 for polynomial regression.\nType 3 for exponential regression.\n")
regtype=int(input('...:'))


X = data['0'].values
Y = data['1'].values


def linear(X, Y):
    X_mat = np.vstack((np.ones(len(X)), X)).T
    coef = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y)
    line_y = X_mat.dot(coef)
    plt.scatter(X, Y)
    plt.title('Linear regression', fontsize=20)
    plt.xlabel(r'$R^2$ score: {0}'.format(r2_score(Y, line_y)))
    plt.plot(X, line_y, color='red', linewidth=4),
    plt.show()


def poly(X, Y):
    X = X.reshape(-1, 1)
    i = 2
    bestscore = -10000
    while True:
        poly = PolynomialFeatures(degree=i)
        x_poly = poly.fit_transform(X)
        poly.fit(x_poly, Y)
        lin = LinearRegression()
        lin.fit(x_poly, Y)
        score = r2_score(Y, lin.predict(x_poly))
        if score > bestscore:
            bestscore = score
            bestmodel = lin
            bestpoly = poly
        if score >= 0.99 or score < -500 or score < bestscore:
            break
        i+=1
        #print(score,i)
    x = np.linspace(min(X), max(X), num=50)
    x_new = bestpoly.fit_transform(x)
    y_pred = bestmodel.predict(x_new)
    plt.scatter(X, Y)
    plt.title('Polynomial regression', fontsize=20)
    plt.xlabel(r'$R^2$ score: {0}'.format(bestscore))
    plt.plot(x, y_pred, color="red")
    plt.show()


def func(x, a, b, c):
    return a * np.exp(-x * b) + c

def exp(X, Y):
    popt, pcov = curve_fit(func, X, Y,p0=(1, 1e-6, 1),maxfev=2000)
    x_plot = np.linspace(min(X), max(X))
    plt.scatter(X, Y)
    y_plot = func(x_plot, *popt)
    plt.plot(x_plot, y_plot, "r-")
    plt.title("Exponential regression")
    plt.xlabel(r'$R^2$ score: {0}'.format(r2_score(Y,func(X, *popt))))
    plt.show()



if regtype == 1:
    linear(X,Y)
elif regtype == 2:
    poly(X,Y)
elif regtype == 3:
    exp(X,Y)
else:
    print("Error 404")
