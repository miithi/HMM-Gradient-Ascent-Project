import numpy as np

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR data
X = np.array([
    [0.6, 0.4],
    [0.1, 0.2],
    [0.8, 0.6],
    [0.3, 0.7],
    [0.7, 0.3],
    [0.7, 0.7],
    [0.2, 0.9]
])

y = np.array([[1], [0], [0], [1], [1], [0], [1]])

# Initialize weights as per the problem
w = np.array([1, 2, -1, 1, -2, 1])

# Reshape to fit the architecture
# Input to hidden weights (2 input features, 2 hidden nodes)
w_input_hidden = np.array([[w[0], w[1]], [w[2], w[3]]], dtype=float)

# Hidden to output weights (2 hidden nodes, 1 output node)
w_hidden_output = np.array([[w[4]], [w[5]]], dtype=float)

# Learning rate
alpha = 0.1

# Number of epochs
epochs = 10000

# Train the network using SGD
for epoch in range(epochs):
    total_loss = 0
    
    for i in range(len(X)):
        # Forward pass
        input_layer = X[i].reshape(1, 2)  # Reshape to row vector (1, 2)
        hidden_layer_input = np.dot(input_layer, w_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        
        output_layer_input = np.dot(hidden_layer_output, w_hidden_output)
        output = sigmoid(output_layer_input)
        
        # Calculate error (loss) for the current sample
        error = y[i] - output
        total_loss += error ** 2
        
        # Backpropagation
        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(w_hidden_output.T) * sigmoid_derivative(hidden_layer_output)
        
        # Update weights using SGD
        w_hidden_output += alpha * hidden_layer_output.T.dot(d_output)
        w_input_hidden += alpha * input_layer.T.dot(d_hidden)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.sum()}')

# Testing the final model after training
def predict(X_test):
    hidden_layer_input = np.dot(X_test, w_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, w_hidden_output)
    output = sigmoid(output_layer_input)
    return output

# Test data from the screenshot
X_test = np.array([
    [0.55, 0.11],
    [0.32, 0.21],
    [0.24, 0.64],
    [0.86, 0.68],
    [0.53, 0.79]
])

# Run predictions
predictions = predict(X_test)
print("Predictions on test set:")
print(predictions)
