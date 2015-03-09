#!/usr/bin/env python

import sys
from numpy import loadtxt,ones,zeros,array
import matplotlib.pyplot as plt

#This code do linear regression 

def computeCost(X,y,theta):
    '''
    computeCost 
    Calculate J
    '''
    J = 0
    #Number of training samples
    m = y.size
    
    #Prediction or hypothesis
    hypothesis = X.dot(theta).flatten()
    sqerr = (hypothesis -y)**2
    J = (1.0/(2*m))*sqerr.sum()

    return J

def gradientDescent(X,y,theta,alpha,iteration):
    '''
    Perform gradient descent to learn theta
    by taking iteration gradient steps with learning
    rate alpha.Here we are calculating only for x0 , x1. Not for multiplefeatures 
    '''
    m = y.size
    J_history = zeros(shape=(iteration,1))  
    
    for i in range(iteration):
        #hypothesis
        hypothesis = X.dot(theta).flatten()

        error_theta0 = (hypothesis - y) * X[:,0]
        error_theta1 = (hypothesis - y) * X[:,1]

        theta[0][0] = theta[0][0] - alpha * (1.0/(2*m))*error_theta0.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0/(2*m))*error_theta1.sum()

        J_history[i,0] = computeCost(X,y,theta)

    return theta,J_history


    
#Loading Data
data = loadtxt('ex1data1.txt',delimiter=',')  # Change delimeter if you have any other delimeter in your file
    
#Plotting Data
#plt.plot(data[:,0],data[:,1],'ro')
#plt.show()
    
X = data[:,0]   #Feature
y = data[:,1]   #Output
m = y.size      #Number of training set

#Add column of x0=1 to X

x0 = ones(shape=(m,2))  #This will create  ones of x0 of m*2 
x0[:,1] = X            #It will add column of 1 and X to x0

#Lets initialize Theta
theta = zeros(shape=(2,1))  #2*1

J = computeCost(x0,y,theta)
print "Compute Cost :",J


#Setting variables for gradient descent
iterations = 1500;
alpha = 0.01

theta , J_history = gradientDescent(x0, y, theta, alpha, iterations);

print "Theta : ",theta

#Predict values for population sizes of 35,000 and 70,000
print "Predict"
predict1 = array([1, 3.5]).dot(theta).flatten()
print predict1
print "For population of 35000 ,the predicted profit is : %f"% (predict1 * 10000)

