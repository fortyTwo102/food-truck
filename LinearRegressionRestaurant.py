import os
import numpy as np
from matplotlib import pyplot

def plotData(x, y,theta):
	fig = pyplot.figure()
	pyplot.plot(x, y, 'ro', ms=10, mec = 'k')
	pyplot.xlabel("X")
	pyplot.ylabel("y")

	if theta[0]!=-9999:
		pyplot.plot(x, np.dot(np.stack([np.ones(y.size), x], axis=1), theta), '-')
		pyplot.legend(['Training data', 'Linear regression']);

	pyplot.show()

def computeCost(X,y, theta):

	m = y.size

	'''hypo = np.dot(X,theta)
	sqDiff = np.square(hypo - y)
	J = 1/(2*m)*np.sum(sqDiff)'''

	J = 1/(2*m)*np.sum(np.square(np.dot(X,theta) - y))

	return J

def plotIteration(itr, J_history):
	itr = np.arange(itr)
	fig = pyplot.figure()
	pyplot.plot(itr, J_history, 'ro', ms=10, mec = 'k')
	pyplot.show()

def gradientDescent(X, y, theta, alpha, num_iters):
	m = y.shape[0]
	theta = theta.copy()

	J_history = []

	for i in range(num_iters):
		diff = np.dot(X, theta) - y
		for i in range(len(theta)):			
			theta[i]= theta[i] - alpha*1/m*(np.dot(diff,X[:, i]))

		J_history.append(computeCost(X, y, theta))
	


	return theta, J_history

def LinearRegressionModel(filepath):

	data = np.loadtxt(filepath, delimiter = ',')

	X,y = data[:, 0], data[:, 1]
	#If you want to check the dimensions, use .shape attribute 
	m = y.size
	#plotData(X,y,[-9999,-9999]) #-9999 for signifying no theta
	X = np.stack([np.ones(m), X], axis=1)

	theta=np.zeros(X.shape[1])
	J = computeCost(X,y, theta)
	print('Cost in case of parameters [0,0]: ',J)

	theta=np.zeros(X.shape[1])
	alpha = 0.01
	num_iters = 1500

	theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
	plotIteration(num_iters, J_history)
	print('Cost : ',computeCost(X,y,theta))
	print('\nTheta found by gradient descent:',theta)


	plotData(X[:, 1], y,theta)

	print('RMSE : ',np.sqrt(np.mean((np.dot(X, theta) - y)**2)))


LinearRegressionModel("ex1data1.txt") #pass filepath of dataset