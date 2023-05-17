import matplotlib.pyplot as plt
#1D Linear Regression

def sum(x):
    s = 0.0
    for v in x:
        s += v
    return s

def sqrt(x):
    return x ** (1/2)

def mean(x):
    return sum(x) / float(len(x))

def variance(x):
    m = mean(x)
    return sum([(v - m)**2 for v in x])

def covariance(x, y):
    covar = 0.0
    mx = mean(x)
    my = mean(y)
    for i in range(len(x)):
        covar += (x[i] - mx) * (y[i] - my)
    return covar

def rmse(actual, pred):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += ((pred[i] - actual[i]) ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)



def linear_regression(x, y):
    # receives observations
    # returns params a,b
    # h(xi) = b0 + b1*xi
    b1 = covariance(x, y) / variance(x)
    b0 = mean(y) - b1*mean(x)    
    return b0, b1
    

# example
x = [1,2,3,4,5,6,7,8,9,10]
y = [2.1,4.2,2.4,6.3,8.6,2.1,7.8,9.9,1.0,2.0]
# compute coeffs
b0, b1 = linear_regression(x,y)
# generate line
x_t = list(range(min(x),max(x)+2))
y_t = [b0+i*b1 for i in x_t]
# compute error
error = rmse(y, y_t)
print(error)
# plot results
plt.scatter(x,y)
plt.plot(y_t)
plt.show()

