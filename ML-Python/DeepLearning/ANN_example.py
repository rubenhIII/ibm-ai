import numpy as np # import Numpy library to generate

def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x)))

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases

x_1 = 0.5 # input 1
x_2 = 0.85 # input 2

print('x1 is {} and x2 is {}'.format(x_1, x_2))

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(z_12))

a_11 = sigmoid(z_11)
a_12 = sigmoid(z_12)
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
a_2 = sigmoid(z_2)
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))