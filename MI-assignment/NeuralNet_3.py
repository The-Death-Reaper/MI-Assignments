'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
	Number of epochs : 200
	Batch Size : 2
	Number of epochs after which Learning Rate decays : 50
	Test Train Split ratio : 0.3
	We perform stratified splitting on the dataset
	We have added layer specific adaptive learning rate
	Number of layers : 1 Input layer:
							10 neurons one for each feature in the dataset
					   2 Hidden Layers:
					   		Hidden layer 1 has 20 neurons.
							   The Learning Rate for the layer is 0.05
							   The activation function used is the tanh function
							Hidden layer 2 has 16 neurons
							   The Learning Rate for the layer is 0.06
							   The activation function used is the tanh function
					   1 Output Layer
					   		1 neuron
							   The Learning Rate for the layer is 0.9
							   The Learning Rate decay is set to 0.005
							   The activation function used is the Sigmoid function
							
						The lambda value for regularisation was set to 0 as the model showed no signs of overfitting
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# number_of_units_layer1 = 9
# number_of_units_layer2 = 10
# number_of_units_layer3 = 1


# sigmoid activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# derivative of sigmoid activation function
def sigmoid_prime(x):
	return np.exp(-x) / (1 + np.exp(-x))**2

# tanh activation function
def tanh(x):
	return np.tanh(x)

# derivative of tanh activation function
def tanh_prime(x):
	return 1 - np.tanh(x)**2

# relu activation function
def relu(x):
	return np.maximum(x, 0)

# derivative of relu activation function
def relu_prime(x):
	return np.array(x >= 0).astype('int')

# mean square error
def mse(y_true, y_pred):
	return np.mean(np.power(y_true - y_pred, 2))

# derivative of mean square error
def mse_prime(y_true, y_pred):
	return 2 * (y_pred - y_true) / y_pred.size

# sum of square errors
def sse(y_true, y_pred):
	# print(y_true - y_pred)
	return 0.5 * np.sum(np.power(y_true - y_pred, 2))

# derivative of sum of square errors
def sse_prime(y_true, y_pred):
	return y_pred - y_true


class Layer:
	def __init__(self, input_connections, no_of_nodes, activation, d_activation, learning_rate, lr_decay = 0, r_lambda = 0):
		

		self.input_connections = input_connections 																		# input size of the layer
		self.activation = activation 																					# activation function of the layer
		self.d_activation = d_activation 																				# derivative of the activation function of the layer
		self.no_of_nodes = no_of_nodes 																					# number of neurons in the layer
		self.weights = np.random.randn(input_connections, no_of_nodes)*np.sqrt(2/(input_connections + no_of_nodes))		# weight matrix of the layer
		self.bias = np.random.randn(1, no_of_nodes)*0.002															# bias matrix of the layer
		self.lr = learning_rate 																						# learning rate corresponding to the layer
		self.lr_decay = lr_decay 																						# decay rate of learning rate
		self.r_lambda = r_lambda 																						# value of lambda for regularization

	#function to perform forward propagation 
	#returns the value obtained on applying the activation function on the summarized value of the input, weights and bias
	def forward_prop(self, x):
		self.input = x #input to the layer

		#value after performing linear combination of the weights and the input along with the bias
		self.input_to_act = np.dot(x, self.weights) + self.bias 

		return self.activation(self.input_to_act)

	 
	#function to perform backward propagation
	#returns the error to be propagated back to the previous layer
	def backward_prop(self, error):

		# error: d(loss)/d(output) [received as an argument to the function]
		# activation_error: d(loss)/d(output) * d(output)/d(summarized_value)
		# weights: d(loss)/d(wi): d(loss)/d(output) * d(output)/d(summarized_value) * d(summarized_value)/d(wi) = activation_error * d(summarized_value)/d(wi) = activation_error * input
		# bias: d(loss)/d(bias) : d(loss)/d(output) * d(output)/d(summarized_value) * d(summarized_value)/d(b) = activation_error
		# input: d(loss)/d(input) : d(loss)/d(summarized_value) * d(summarized_value)/d(in) = activation_error * wi

		activation_error = error * (self.d_activation(self.input_to_act))
		
		error_to_back_prop = np.dot(activation_error, self.weights.T)

		reg = (self.r_lambda/(2*activation_error.shape[0]))*self.weights 
		self.weights -= (np.dot(self.input.T, activation_error)*self.lr + reg)
		self.bias -= (1/activation_error.shape[0]) * np.sum(activation_error)*self.lr
		return error_to_back_prop



class NN:

	''' X and Y are dataframes '''

	def __init__(self, epochs, batch_size):
		self.layers = [] 				# list of layers of the neural network
		self.loss = mse 				# loss function for the neural network
		self.d_loss = mse_prime 		# deriavtive of the loss function
		self.epochs = epochs 			# number of epochs
		self.batch_size = batch_size 	# sets the batch size for the model

	# add layer to network
	def add(self, layer):
		self.layers.append(layer)

	
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
		iterations = int(len(X)/self.batch_size)
		for e in range(self.epochs):
			loss = 0
			k = 0

			for i in range(iterations):
				output = X[k:k + self.batch_size]

				for layer in self.layers:
					output = layer.forward_prop(output)
				
				Y_true = Y[k:k + self.batch_size]
				k += self.batch_size
				Y_true = Y_true.reshape(-1, 1)
				loss += self.loss(Y_true, output)
				error = self.d_loss(Y_true, output)

				for layer in reversed(self.layers):
					error = layer.backward_prop(error)
					if(e%50==0):
						layer.lr *= (1 / (1 + layer.lr_decay * e))
			print('epoch %d/%d   error=%f' % (e+1, self.epochs, loss/iterations))

	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values

		yhat is a list of the predicted value for df X
		"""
		y_hat = X

		for layer in self.layers:
			y_hat = layer.forward_prop(y_hat)
		return y_hat

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
		# print(y_test, y_test_obs)
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
				fn=fn+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fp=fp+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p = tp/(tp+fp)
		r = tp/(tp+fn)
		f1 = (2*p*r)/(p+r)
		ac = (tp+tn)/(tp+tn+fp+fn)
		print("Confusion Matrix : ", end="")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print(f"ACCURACY : {ac}")

def load_data():
	dataset = 'LBW_Dataset_clean.csv'
	df = pd.read_csv(dataset)
	df1 = df.drop(columns=['Result'])
	df2 = df['Result']
	df1 = df1.to_numpy()
	df2 = df2.to_numpy()
	return (df1, df2)
	
X,Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=32, stratify=Y, shuffle=True)

np.random.seed(14)
nn = NN(epochs = 200, batch_size = 2)
nn.add(Layer(10, 20, tanh, tanh_prime, learning_rate = 0.05, lr_decay = 0, r_lambda = 0))
nn.add(Layer(20, 16, tanh, tanh_prime, learning_rate = 0.06, lr_decay = 0, r_lambda = 0))
nn.add(Layer(16, 1, sigmoid, sigmoid_prime, learning_rate = 0.9, lr_decay = 0.005, r_lambda = 0))
nn.fit(X_train, Y_train)
print()
print("-"*150)
print("\nPERFORMANCE METRICS FOR TRAINING: \n")
NN.CM(Y_train, nn.predict(X_train))
print()
print("-"*150)
print("\nPERFORMANCE METRICS FOR TESTING: \n")
NN.CM(Y_test, nn.predict(X_test))
print()
print("-"*150)
print()
