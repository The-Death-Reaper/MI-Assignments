'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
	Number of layers : 1 Input layer (9 neurons)
					   2 Hidden Layers (11 neurons & 10 neurons)
					   1 Output Layer  (1 neuron)
'''
import numpy as np
import pandas as pd
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split

number_of_units_layer1 = 9
number_of_units_layer2 = 10
number_of_units_layer3 = 1

EPOCHS = 20000
BATCH_SIZE = 8
learning_rate = 0.01

def sigmoid(x):
	z = 1/(1 + np.exp(-x))
	return z

def sigmoid_derivtive(x):
	return sigmoid(x) * (1 - sigmoid(x))

def relU(x):
	return x * (x > 0)

class NN:

	''' X and Y are dataframes '''
	def __init__(self):
		self.n_x = number_of_units_layer1
		self.n_hidden = number_of_units_layer2
		self.n_out = number_of_units_layer3
		#Initialising layer 1
		self.W1 = np.random.randn(self.n_hidden,self.n_x) * 0.01
		self.b1 = np.zeros(shape=(self.n_hidden, 1))
		#Initialising layer 2
		self.W2 = np.random.randn(self.n_out,self.n_hidden) * 0.01
		self.b2 = np.zeros(shape=(self.n_out, 1))

		self.dZ2 = 0
		self.dW2 = 0
		self.db2 = 0
		self.dZ1 = 0
		self.dW1 = 0
		self.db1 = 0


	def loss(self,y_pred,y_obs):
		cross_entropy = np.multiply(np.log(y_pred),y_obs) + np.multiply(1 - y_obs, np.log(1 - y_pred))
		cost = - np.sum(cross_entropy) / len(y_pred)
		return cost

	def back_prop(self,  X, Y):
		""" Back-progagate gradient of the loss """
		m = X.shape[0]
		self.dZ2 = self.A2 - Y
		self.dW2 = (1 / m) * np.dot(self.dZ2, self.A1.T)
		self.db2 = (1 / m) * np.sum(self.dZ2, axis=1, keepdims=True)
		self.dZ1 = np.multiply(np.dot(self.W2.T, self.dZ2), 1 - np.power(self.A1, 2))
		self.dW1 = (1 / m) * np.dot(self.dZ1, X)
		self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)


	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
		m = X.shape[0]
		for e in range(EPOCHS):
			self.forward(X)
			loss = -np.sum(np.multiply(np.log(self.A2), Y) + np.multiply(np.log(1-self.A2),  (1 - Y))) / m
			self.back_prop(X, Y)

			self.W1 -= learning_rate * self.dW1
			self.b1 -= learning_rate * self.db1
			self.W2 -= learning_rate * self.dW2
			self.b2 -= learning_rate * self.db2
			if(e%100 == 1):
				print(e, loss)


	def forward(self,X):
		self.Z1 = self.W1.dot(X.T) + self.b1
		self.A1 = np.tanh(self.Z1)
		self.Z2 = self.W2.dot(self.A1) + self.b2
		self.A2 = sigmoid(self.Z2)


	# def predict_for_single_row(self,x,pass_type="train"):
	# 	Z1 = np.dot(self.W1,x) + self.b1
	# 	A1 = np.tanh(Z1)

	# 	Z2 = np.dot(self.W2,A1) + self.b2
	# 	A2 = sigmoid(Z2)

	# 	return A2

	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values

		yhat is a list of the predicted value for df X
		"""
		# y_hat = []
		# y_hat = X.apply(self.predict_for_single_row,axis = 1)
		self.forward(X)
		return np.round(self.A2).astype(np.int)

	def CM(y_test,y_test_obs):
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
		ac = (tp+tn)/(tp+tn+fp+fn)
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print(f"ACCURACY : {ac}")


def load_data():
	dataset = 'LBW_Dataset_clean.csv'
	df = pd.read_csv(dataset)
	df1 = df.drop(columns='Result')
	df2 = df['Result']
	df1 = df1.to_numpy()
	df2 = df2.to_numpy()
	return (df1, df2)
	pass
X,Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

nn = NN()
nn.fit(X_train,Y_train)
NN.CM(Y_test, nn.predict(X_test)[0])
print(nn.predict(X_test), Y_test)

# 32