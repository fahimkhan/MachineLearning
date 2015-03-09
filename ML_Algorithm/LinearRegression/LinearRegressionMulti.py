#!/usr/bin/env python

from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

"""
Before going directly to the linear regression it is important to analyze our data. 
By looking at the values, note that house sizes are about 1000 times the number of bedrooms. 
When features differ by orders of magnitude, it is important to perfom a feature scaling that can make gradient descent converge much more quickly.
"""


print "Starting Linear regression"


def feature_normalize(X):
    print "Do normalization of features"
    mean_r = []
    std_r = []
    X_norm = X

    no_of_column = X.shape[1]  #To get number of zero use 0

    for i in range(no_of_column):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s
          
    return X_norm, mean_r, std_r 

def computeCost(X,y,theta):
    '''
    computeCost 
    Calculate J
    '''
    J = 0
    m = y.size

    hypothesis = X.dot(theta).flatten()
    sqerr = (hypothesis -y)**2
    J = (1.0/(2*m))*sqerr.sum()

    return J



def gradient_descent_multi(X, y, theta, alpha, num_iters):
    print "Doing gradient descent on final data"
    
    m = y.size
    J_history = zeros(shape=(num_iters, 1))
      
    for i in range(num_iters):
        predictions = X.dot(theta)
        theta_size = theta.size  #Finding number of features
        
        for item in range(theta_size):
            temp = X[:, item]
            temp.shape = (m, 1)
              
            errors_x1 = (predictions - y) * temp
            theta[item][0] = theta[item][0] - alpha * (1.0 / m) * errors_x1.sum()
                
        J_history[i, 0] = computeCost(X, y, theta)

    return theta, J_history 


#Loading data for analysis
data = loadtxt('ex1data2.txt',delimiter=',') #Change delimeter here

#Plotting Data
"""
fig = plt.figure()
table = fig.add_subplot(111,projection='3d')
n=100
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    table.scatter(xs, ys, zs, c=c, marker=m)
    
table.set_xlabel('Size of the House')
table.set_ylabel('Number of Bedrooms')
table.set_zlabel('Price of the House')
plt.show()
"""

X = data[:,:2]  # Taking first two features x1 and x2
y = data[:,2]   # Output

#Number of training samples
m = y.size

#First scale feature and do normalization of our data set or features
x,mean_r,std_r = feature_normalize(X)

#print "Mean :",mean
#print "Std :",std
#print "Normalize X :",x

#Add x0 to normalize data
x0 = ones(shape=(m, 3)) 
x0[:, 1:3] = x

#Some gradient descent settings
iterations = 1000
alpha = 0.001
 
#Init Theta and Run Gradient Descent
theta = zeros(shape=(3, 1)) 
theta,J_history = gradient_descent_multi(x0, y, theta, alpha, iterations)

#print "Theta : ",theta

#Predict price of a 1650 sq-ft 3 br house
price = array([1.0, ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house: %f' % (price) 




