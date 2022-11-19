import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)*0.05
        # print("HERERE", self.weights, "here")
        self.bias = np.zeros(shape=(1, output_size))

    def forward(self, input):
        self.input = input
        # print(input, self.weights)
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        # print(output_error)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error
        
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)

# bonus
class FlattenLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.reshape(input, (1, -1))
    
    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)

# bonus
class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_true - y_pred, 2))

def sse_prime(y_true, y_pred):
    return y_pred - y_true
def loss(y_pred,y_obs):
    cross_entropy = np.multiply(np.log(y_pred),y_obs) + np.multiply(1 - y_obs, np.log(1 - y_pred))
    cost = - np.sum(cross_entropy) / len(y_pred)
    return cost

from sklearn.model_selection import train_test_split
import pandas as pd


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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)
# print(x_train)
# unlike the Medium article, I am not encapsulating this process in a separate class
# I think it is nice just like this
network = [
    FlattenLayer(9),
    FCLayer(9, 10),
    ActivationLayer(tanh, tanh_prime),
    FCLayer(10, 1),
    ActivationLayer(sigmoid, sigmoid_prime),
]

epochs = 2000
learning_rate = 0.02

# training
for epoch in range(epochs):
    error = 0
    for x, y_true in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)
        # output = np.round(output-0.1).astype(np.int)
        # print(output, y_true)
        # error (display purpose only)
        error += sse(y_true, output)

        # backward
        output_error = sse_prime(y_true, output)
        for layer in reversed(network):
            output_error = layer.backward(output_error, learning_rate)
    
    error /= len(x_train)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    # output = np.round(output-0.1).astype(np.int)
    return output
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
l=[]
for i in x_test:
    # print(i, type(i))
    l.append(predict(network, i)[0][0])
print(l, y_test)
CM(y_test, np.array(l))
ratio = sum([np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
error = sum([mse(y, predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print('ratio: %.2f' % ratio)
print('mse: %.4f' % error)

# import matplotlib.pyplot as plt

# samples = 10
# for test, true in zip(x_test[:samples], y_test[:samples]):
#     image = np.reshape(test, (9,9))
#     plt.imshow(image, cmap='binary')
#     plt.show()
#     pred = predict(network, test)[0]
#     idx = np.argmax(pred)
#     idx_true = np.argmax(true)
#     print('pred: %s, prob: %.2f, true: %d' % (idx, pred[idx], idx_true))