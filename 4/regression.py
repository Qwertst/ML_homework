import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

data = pd.read_csv("data.txt")
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
    plt.xlabel('MSE: {0}'.format(mean_squared_error(Y, line_y)))
    #plt.xlabel(r'$R^2$ score: {0}'.format(r2_score(Y, line_y)))
    plt.plot(X, line_y, color='red', linewidth=4)
    plt.savefig('lin3.png')
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
        if score >= 0.99 or i>10 or score < bestscore:
            break
        i+=1
    x = np.linspace(min(X), max(X), num=50)
    x_new = bestpoly.fit_transform(x)
    y_pred = bestmodel.predict(x_new)
    plt.scatter(X, Y)
    plt.title('Polynomial regression', fontsize=20)
    #plt.xlabel(r'$R^2$ score: {0}'.format(bestscore))
    plt.xlabel('MSE: {0}'.format(mean_squared_error(Y, bestmodel.predict(bestpoly.fit_transform(X)))))
    plt.plot(x, y_pred, color="red",linewidth=4)
    plt.savefig('poly3.png')
    plt.show()


def func(x, a, b, c):
    return a * np.exp(-x * b) + c

def exp(X, Y):
    popt, pcov = curve_fit(func, X, Y,p0=(-1, 1, 1),maxfev=2000)
    x_plot = np.linspace(min(X), max(X))
    plt.scatter(X, Y)
    y_plot = func(x_plot, *popt)
    plt.plot(x_plot, y_plot, "r-",linewidth=4)
    plt.title("Exponential regression")
    #plt.xlabel(r'$R^2$ score: {0}'.format(r2_score(Y,func(X, *popt))))
    plt.xlabel('MSE: {0}'.format(mean_squared_error(Y, func(X, *popt))))
    plt.savefig('exp3.png')
    plt.show()



if regtype == 1:
    linear(X,Y)
elif regtype == 2:
    poly(X,Y)
elif regtype == 3:
    exp(X,Y)
else:
    print("Error 404")
