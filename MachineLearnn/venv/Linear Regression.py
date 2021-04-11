from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_gradient_at_b(x, y, b, m):
    N = len(x)
    diff = 0
    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        diff += (y_val - ((m * x_val) + b))
    b_gradient = -(2 / N) * diff
    return b_gradient


def get_gradient_at_m(x, y, b, m):
    N = len(x)
    diff = 0
    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        diff += x_val * (y_val - ((m * x_val) + b))
    m_gradient = -(2 / N) * diff
    return m_gradient


# Your step_gradient function here
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]


# Your gradient_descent function here:
def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0

    for i in range(num_iterations):
        [b, m] = step_gradient(b, m, x, y, learning_rate)
    return [b, m]


# months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]
#
# # Uncomment the line below to run your gradient_descent function
# [b, m] = gradient_descent(months, revenue, 0.01, 1000)
#
# # Uncomment the lines below to see the line you've settled upon!
# y = [m * x + b for x in months]
#
# line_fitter = LinearRegression()
#
# months_array = np.array(months).reshape(-1,1)
# line_fitter.fit(months_array, revenue)
# revenue_predict = line_fitter.predict(months_array)
#
# plt.plot(months, revenue, "o")
# plt.plot(months, y)
# plt.plot(months, revenue_predict)
#
# plt.show()


df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# reset_index() create data frame
prod_per_year = (df.groupby('year').totalprod.mean().reset_index())

# can call column
X = prod_per_year.year
# reshape(-1, 1) rotate columns of array to rows each with one column
X = X.values.reshape(-1, 1)

y = prod_per_year.totalprod.values.reshape(-1,1)

plt.scatter(X, y)
plt.xlabel('Year')
plt.ylabel('Production Per Year')

regr = linear_model.LinearRegression()
regr.fit(X, y)

y_predict = regr.predict(X)
plt.plot(X, y_predict)
plt.show()

X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
print(future_predict[X_future == 2050])
plt.figure()
plt.plot(X_future, future_predict)
plt.show()

