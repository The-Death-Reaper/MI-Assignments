Assignment 3

TeamName : PESU-MI_1439_1452_1456

Team Members :
Mayur P L, PES1201801439
Manu M Bhat, PES1201801452
Darshan D, PES1201801456


Implementation Details:

Data Pre-Processing:

- We observe that the Community column in the dataset has categorical values as the numbers (1,2,3,4) represent different communities. To handle this, we split the Community column into 4 different columns each representing one community. Hence, we perform one hot encoding to handle the categorical attribute Community.

- We observe that the Education column either has a value of 5 or a NaN. Thus the Education column is dropped as it does not provide any significant information.

- We observe that the Delivery Phase column has majority of its values as 1. There are only 2 rows with the value 2 but they have opposing results. Thus, the column is dropped as it has less variance.

- We then handle the outliers by using interquantile range.

- We observe that the Age and Weight columns have a higher range of values as compared to the other columns. Therefore we normalize the values by performing min max normalization to ensure that all the values are between 0 and 1. This would also take care of bringing the binary values 1 and 2 of Residence to the binary values of 0 and 1.

- Filling the NaNs: We fill NaNs of Age, Weight with median value, NaNs of HB, BP with average values, NaNs of Residence with mode.


Building the Neural Network:

- First we define a class to represent a layer in the neural network. This class Layer has attributes such as number of neurons in the layer, number of incoming edges, activation function used in the layer, learning rate, lambda value used for regularization, rate of decay of learning rate, weights and bias initialization. The weights are initialized to random values using the randn function. Since we use tanh as the activation function, we follow the Xavier Initialization and multiply these random weights with the square root of (2/(number of incoming edges + number of neurons in the layer)). Bias is initialized with random values generated using the rand function. These random values are multiplied by a factor of 0.002.
The class Layer has functions to perform forward propagation and backward propagation corresponding to the layer.

- Then we use the template provided to us and add functions and attributes to this class NN. This class NN has attributes such as a list of layer objects, loss function used, number of epochs, batch size. This class has a function add to add a layer to the neural network, a function fit that trains the neural network. The function fit performs forward propagation in batches of size mentioned. For each of these forward propagations, the Mean Squared error value is calculated. Then backward propagation is performed after which the weights and bias are updated accordingly. The class NN also consists of a function predict that only performs forward propagation on the given test data and finds the corresponding result. The function CM is used to find the confusion matrix and calculate the performance metrics for the Neural Network such as Accuracy, Precision, Recall, F1 Score.

- The neural network performs batch gradient descent [batch size = 2]. The loss function used is mean square error.


Hyperparameters:

Number of epochs : 200
Batch Size : 2
Number of epochs after which Learning Rate decays : 50
Test Train Split ratio : 0.3
We perform stratified splitting on the dataset
We have added layer specific adaptive learning rate
Number of layers : 1 Input layer:
                        10 neurons one for each feature in the processed dataset
                   2 Hidden Layers:
                        Hidden layer 1 has 20 neurons.
                           The Learning Rate for the layer is 0.05
                           The activation function used is the tanh function
                        Hidden layer 2 has 16 neurons
                           The Learning Rate for the layer is 0.06
                           The activation function used is the tanh function
                   1 Output Layer:
                        1 neuron
                           The Learning Rate for the layer is 0.9
                           The Learning Rate decay is set to 0.005
                           The activation function used is the Sigmoid function
The dimensions for each layer would be (size_l, size_l-1) where size_l is the number of neurons in the current layer
The bias for each layer would be (1, size_l) where size_l is the number of neurons in the current layer
Mean Squared Error loss function was used
Decay for Learning Rate was added so that while performing gradient descent we converge to the global minima without overshooting.
Regularisation was implemented to prevent overfitting on training data.
The lambda value for regularisation was set to 0 as the model showed no signs of overfitting

Key Features of the design:

1) We have defined a class to represent a Layer. This class has functions to perform forward propagation and backward propagation. The class Layer has attributes such as number of neurons in the layer, activation function used in the layer, learning rate, lambda value used for regularization, rate of decay of learning rate, weights and bias. By making Layer a class, addition of new layers with different set of paramaters can be done by just creating objects with the appropriate attributes initialized.

2) The modularity and use of Object Oriented Programming concepts makes the design of the neural network generic and ensures that it can be used for different applications by providing the appropriate parameters.

3) Layer Specific Adaptive Learning Rate: Learning rates are defined separately for each layer. Therefore, we can use different learning rates that increase as we move towards the output layer, which has been proven to give better performance. We also provide specific decay rates for the learning rates in each layer.

4) To prevent overfitting, we perform regularization by using appropriate values of lambda.



Implementing something beyond the basics:

Yes, we have implemented quite a few things that are beyond the basics.
- We have implemented regularization to prevent overfitting.
- Layer Specific Dynamic Learning Rate.
- We have assigned different learning rates for different layers because assigning progressively increasing learning rates as we move towards the output layer has shown to give better performance.
- We also use the Xavier method of initializing the weights since we use the tanh activation function.
- We also define a decay rate that will decay the learning rate to ensure that a global minima is reached.


Detailed steps to run our files:

The data folder contains the cleaned dataset.

The src folder contains the cleandata.py that performs the pre processing on the dataset.
To run the cleandata.py file, give the command python3 cleandata.py

The src folder also contains the neural_net.py which contains the source code that builds the neural network.
To run the neural_net.py file, give the command python3 neural_net.py
