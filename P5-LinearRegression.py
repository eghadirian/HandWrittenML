def predict(alpha, beta, x_i):
    return beta*x_i+alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i)**2 for x_i, y_i in zip(x,y))

def mean(x):
    return sum(x)/len(x)

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def dot(v,w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot(v,v)

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations)/(n-1)

def standard_deviation(x):
    return np.sqrt(variance(x))

def covariance(x,y):
    n = len(x)
    return dot(de_mean(x), de_mean(y))/(n-1)

def correlation(x,y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x,y)/stdev_x/stdev_y
    else:
        return 0

def least_square_fit(x, y):
    beta = correlation(x,y)*standard_deviation(y)/standard_deviation(x)
    alpha = mean(y) - beta*mean(x)
    return alpha, beta

def total_sum_of_squares(y):
    return sum(v**2 for v in de_mean(y))

def r_squared(alpha, beta, x, y):
    return 1.-(sum_of_squared_errors(alpha, beta, x, y)/total_sum_of_squares(y))

