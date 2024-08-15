import numpy as np
class NeuralNetwork:
    """
    Neural Network Class
    
    """
    def __init__(self, input_size, hidden_size, output_size = 1, dropout_rate = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x*(1-x)
    
    def forward(self, x, training = False):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        if training:
            self.mask_dropout = np.random.binomial(1, 1 - self.dropout_rate)/(1 - self.dropout_rate)
            self.a1 *= self.mask_dropout
        self.z2 = np.dot(self.z1,self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return(self.a2)
    
    def backward(self, x, y, output, lr = 0.1):
        da2 = y - output
        self.delta2 = da2 * self.sigmoid_derivative(output)
        self.error_hidden = np.dot(self.delta2, self.W2.T)
        self.delta1 = self.error_hidden * self.sigmoid_derivative(self.a1) * self.mask_dropout
        self.W2 += np.dot(self.a2.T, self.delta2)
        self.b2 += np.sum(self.delta2, axis = 0, keepdims = True)
        self.W1 += np.dot(x.T, self.delta1)
        self.b1 += np.sum(self.delta1, axis = 0, keepdims = True)
        return da2

    def train(self, X, y, epochs, lr = 0.01):
        history = []
        for _ in range(epochs):
            output = self.forward(X, training = True)
            history.append(self.backward(X, y, output, lr = lr))
        print(history)
            

        
