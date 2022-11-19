import numpy as np
import pandas as pd

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

class NN:

	''' X and Y are dataframes '''
	def fit(self, X, Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''

		# Sigmoid activation
		def activation(self, z):
			return np.reciprocal(1 + np.exp(-z))
		
		# Binary cross entropy loss function
		def cost(self, X, Y, theta, l=0):
			m = Y.size
			grad = np.zeros(theta.shape[0])

			htheta = activation(self, np.matmul(X, theta))
			rgl = l/(2*m) * np.dot(theta[2:], theta[2:])
			J = (np.dot(Y, np.log(htheta)) + np.dot(1- Y, np.log(htheta)))/-m + rgl
			
			grad[0] = np.dot(X[:, 1], htheta - Y) / m
			grad[1:] = np.matmul(X[:, 2:].T, htheta - Y)/m + (l/m)*theta[1:]

			return (J, grad)

		# Cheapo grad descent for the lack of the ability to use scipy optimizers
		def gradDescent(self, X, Y, theta, alpha, num_iters):
			m = Y.size
			for _ in range(num_iters):
				htheta = activation(self, np.matmul(X, theta))
				theta -= (alpha/m) * np.sum((htheta - Y)*X, 0).T
	
	def predict(self, X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		
		return yhat

	def CM(self, y_test, y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
