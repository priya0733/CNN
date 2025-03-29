"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import NetworkBuildAnimation from "@/components/animations/network-build-animation"
import NetworkTrainingAnimation from "@/components/animations/network-training-animation"
import NetworkEvaluationAnimation from "@/components/animations/network-evaluation-animation"
import CodeDownloader from "@/components/code-downloader"

export default function HandsOnPage() {
  const [activeTab, setActiveTab] = useState("build")

  const buildCode = `
# Building a neural network from scratch
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with the given layer sizes.
        
        Args:
            layer_sizes: List of integers, where each integer represents the number of neurons in a layer.
                         The first element is the input size, the last is the output size.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(1, self.num_layers):
            # He initialization for weights
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            
        Returns:
            activations: List of activations for each layer
            z_values: List of weighted inputs for each layer
        """
        activations = [X]  # List to store activations of each layer
        z_values = []      # List to store z values (weighted inputs) of each layer
        
        # Forward propagation
        for i in range(self.num_layers - 1):
            # Calculate weighted input
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function (sigmoid for output layer, ReLU for hidden layers)
            if i == self.num_layers - 2:  # Output layer
                activation = self.sigmoid(z)
            else:  # Hidden layers
                activation = self.relu(z)
            
            activations.append(activation)
        
        return activations, z_values

# Example usage
layer_sizes = [2, 4, 1]  # 2 inputs, 4 hidden neurons, 1 output
nn = NeuralNetwork(layer_sizes)

# Test forward propagation with sample input
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
activations, z_values = nn.forward(X)

print("Input:")
print(X)
print("\\nOutput:")
print(activations[-1])
`

  const trainCode = `
# Training the neural network
import numpy as np

class NeuralNetwork:
    # ... (previous code from build step) ...
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels, shape (batch_size, output_size)
            y_pred: Predicted values, shape (batch_size, output_size)
            
        Returns:
            loss: The binary cross-entropy loss
        """
        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, X, y, activations, z_values):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            y: True labels, shape (batch_size, output_size)
            activations: List of activations from forward pass
            z_values: List of z values from forward pass
            
        Returns:
            gradients: Dictionary containing gradients for weights and biases
        """
        m = X.shape[0]  # Batch size
        gradients = {"weights": [], "biases": []}
        
        # Initialize backpropagation with output layer error
        dA = activations[-1] - y  # Derivative of loss with respect to output activation
        
        # Backpropagate through each layer
        for l in reversed(range(self.num_layers - 1)):
            if l == self.num_layers - 2:  # Output layer
                dZ = dA * self.sigmoid_derivative(activations[l+1])
            else:  # Hidden layers
                dZ = dA * self.relu_derivative(activations[l+1])
            
            # Compute gradients
            dW = np.dot(activations[l].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Store gradients (in reverse order)
            gradients["weights"].insert(0, dW)
            gradients["biases"].insert(0, db)
            
            # Compute dA for next layer (if not input layer)
            if l > 0:
                dA = np.dot(dZ, self.weights[l].T)
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """
        Update weights and biases using gradients.
        
        Args:
            gradients: Dictionary containing gradients for weights and biases
            learning_rate: Learning rate for gradient descent
        """
        for l in range(self.num_layers - 1):
            self.weights[l] -= learning_rate * gradients["weights"][l]
            self.biases[l] -= learning_rate * gradients["biases"][l]
    
    def train(self, X, y, epochs, learning_rate, verbose=True):
        """
        Train the neural network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            y: True labels, shape (batch_size, output_size)
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            verbose: Whether to print progress
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            activations, z_values = self.forward(X)
            y_pred = activations[-1]
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward propagation
            gradients = self.backward(X, y, activations, z_values)
            
            # Update parameters
            self.update_parameters(gradients, learning_rate)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = np.array([[0], [1], [1], [0]])              # XOR outputs

layer_sizes = [2, 4, 1]  # 2 inputs, 4 hidden neurons, 1 output
nn = NeuralNetwork(layer_sizes)

# Train the network
losses = nn.train(X, y, epochs=1000, learning_rate=0.1)

# Check final predictions
activations, _ = nn.forward(X)
predictions = activations[-1]
print("\\nFinal predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")
`

  const evaluateCode = `
# Evaluating the neural network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NeuralNetwork:
    # ... (previous code from build and train steps) ...
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using the trained network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            threshold: Threshold for binary classification
            
        Returns:
            predictions: Binary predictions
        """
        activations, _ = self.forward(X)
        y_pred = activations[-1]
        return (y_pred >= threshold).astype(int)
    
    def evaluate(self, X, y_true, threshold=0.5):
        """
        Evaluate the model performance.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            y_true: True labels, shape (batch_size, output_size)
            threshold: Threshold for binary classification
            
        Returns:
            metrics: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X, threshold)
        
        # Flatten arrays for sklearn metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true_flat, y_pred_flat),
            "precision": precision_score(y_true_flat, y_pred_flat, zero_division=0),
            "recall": recall_score(y_true_flat, y_pred_flat, zero_division=0),
            "f1": f1_score(y_true_flat, y_pred_flat, zero_division=0)
        }
        
        return metrics
    
    def visualize_decision_boundary(self, X, y, h=0.01):
        """
        Visualize the decision boundary for 2D input data.
        
        Args:
            X: Input data, shape (batch_size, 2)
            y: True labels, shape (batch_size, 1)
            h: Step size for the mesh grid
        """
        # Set min and max values with some padding
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        
        # Create a mesh grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Predict class for each point in the mesh
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3)
        
        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', marker='o', s=100)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = np.array([[0], [1], [1], [0]])              # XOR outputs

# Create and train the network
layer_sizes = [2, 8, 1]  # 2 inputs, 8 hidden neurons, 1 output
nn = NeuralNetwork(layer_sizes)
losses = nn.train(X, y, epochs=2000, learning_rate=0.1)

# Evaluate the model
metrics = nn.evaluate(X, y)
print("\\nModel Evaluation:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Visualize decision boundary
nn.visualize_decision_boundary(X, y)
`

  const fullCode = `
# Complete neural network implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with the given layer sizes.
        
        Args:
            layer_sizes: List of integers, where each integer represents the number of neurons in a layer.
                         The first element is the input size, the last is the output size.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(1, self.num_layers):
            # He initialization for weights
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            
        Returns:
            activations: List of activations for each layer
            z_values: List of weighted inputs for each layer
        """
        activations = [X]  # List to store activations of each layer
        z_values = []      # List to store z values (weighted inputs) of each layer
        
        # Forward propagation
        for i in range(self.num_layers - 1):
            # Calculate weighted input
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function (sigmoid for output layer, ReLU for hidden layers)
            if i == self.num_layers - 2:  # Output layer
                activation = self.sigmoid(z)
            else:  # Hidden layers
                activation = self.relu(z)
            
            activations.append(activation)
        
        return activations, z_values
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels, shape (batch_size, output_size)
            y_pred: Predicted values, shape (batch_size, output_size)
            
        Returns:
            loss: The binary cross-entropy loss
        """
        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, X, y, activations, z_values):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            y: True labels, shape (batch_size, output_size)
            activations: List of activations from forward pass
            z_values: List of z values from forward pass
            
        Returns:
            gradients: Dictionary containing gradients for weights and biases
        """
        m = X.shape[0]  # Batch size
        gradients = {"weights": [], "biases": []}
        
        # Initialize backpropagation with output layer error
        dA = activations[-1] - y  # Derivative of loss with respect to output activation
        
        # Backpropagate through each layer
        for l in reversed(range(self.num_layers - 1)):
            if l == self.num_layers - 2:  # Output layer
                dZ = dA * self.sigmoid_derivative(activations[l+1])
            else:  # Hidden layers
                dZ = dA * self.relu_derivative(activations[l+1])
            
            # Compute gradients
            dW = np.dot(activations[l].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Store gradients (in reverse order)
            gradients["weights"].insert(0, dW)
            gradients["biases"].insert(0, db)
            
            # Compute dA for next layer (if not input layer)
            if l > 0:
                dA = np.dot(dZ, self.weights[l].T)
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """
        Update weights and biases using gradients.
        
        Args:
            gradients: Dictionary containing gradients for weights and biases
            learning_rate: Learning rate for gradient descent
        """
        for l in range(self.num_layers - 1):
            self.weights[l] -= learning_rate * gradients["weights"][l]
            self.biases[l] -= learning_rate * gradients["biases"][l]
    
    def train(self, X, y, epochs, learning_rate, verbose=True):
        """
        Train the neural network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            y: True labels, shape (batch_size, output_size)
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            verbose: Whether to print progress
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            activations, z_values = self.forward(X)
            y_pred = activations[-1]
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward propagation
            gradients = self.backward(X, y, activations, z_values)
            
            # Update parameters
            self.update_parameters(gradients, learning_rate)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using the trained network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            threshold: Threshold for binary classification
            
        Returns:
            predictions: Binary predictions
        """
        activations, _ = self.forward(X)
        y_pred = activations[-1]
        return (y_pred >= threshold).astype(int)
    
    def evaluate(self, X, y_true, threshold=0.5):
        """
        Evaluate the model performance.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            y_true: True labels, shape (batch_size, output_size)
            threshold: Threshold for binary classification
            
        Returns:
            metrics: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X, threshold)
        
        # Flatten arrays for sklearn metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true_flat, y_pred_flat),
            "precision": precision_score(y_true_flat, y_pred_flat, zero_division=0),
            "recall": recall_score(y_true_flat, y_pred_flat, zero_division=0),
            "f1": f1_score(y_true_flat, y_pred_flat, zero_division=0)
        }
        
        return metrics
    
    def visualize_decision_boundary(self, X, y, h=0.01):
        """
        Visualize the decision boundary for 2D input data.
        
        Args:
            X: Input data, shape (batch_size, 2)
            y: True labels, shape (batch_size, 1)
            h: Step size for the mesh grid
        """
        # Set min and max values with some padding
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        
        # Create a mesh grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Predict class for each point in the mesh
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3)
        
        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', marker='o', s=100)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()

# Example: Solving the XOR problem
if __name__ == "__main__":
    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    y = np.array([[0], [1], [1], [0]])              # Outputs
    
    # Create and train the network
    layer_sizes = [2, 8, 1]  # 2 inputs, 8 hidden neurons, 1 output
    nn = NeuralNetwork(layer_sizes)
    
    print("Training the neural network...")
    losses = nn.train(X, y, epochs=2000, learning_rate=0.1)
    
    # Evaluate the model
    metrics = nn.evaluate(X, y)
    print("\\nModel Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Check final predictions
    activations, _ = nn.forward(X)
    predictions = activations[-1]
    print("\\nFinal predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Visualize decision boundary
    nn.visualize_decision_boundary(X, y)
`

  return (
    <div className="container mx-auto space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Hands-on: Build a Simple Neural Network</h1>
        <p className="text-muted-foreground">Step-by-step guide to implement your own neural network from scratch</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="build">Build the Network</TabsTrigger>
          <TabsTrigger value="train">Train the Network</TabsTrigger>
          <TabsTrigger value="evaluate">Evaluate the Network</TabsTrigger>
        </TabsList>

        <TabsContent value="build" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Building a Neural Network</h2>
            <p className="mb-4">
              In this section, we'll implement a neural network from scratch using NumPy. We'll start by defining the
              network architecture and implementing the forward propagation.
            </p>
            <p className="mb-4">Key components we'll implement:</p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Network Initialization:</strong> Define the network architecture and initialize weights and
                biases
              </li>
              <li>
                <strong>Activation Functions:</strong> Implement sigmoid and ReLU activation functions
              </li>
              <li>
                <strong>Forward Propagation:</strong> Implement the forward pass to compute network outputs
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <NetworkBuildAnimation />
              </div>
            </div>
            <p>
              The interactive visualization above demonstrates how a neural network is constructed layer by layer. You
              can adjust the number of layers and neurons to see how the network architecture changes.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="train" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Training the Neural Network</h2>
            <p className="mb-4">
              Now that we've built our neural network, we need to train it using backpropagation and gradient descent.
              This process involves computing the loss, calculating gradients, and updating the weights and biases.
            </p>
            <p className="mb-4">Key components we'll implement:</p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Loss Function:</strong> Implement binary cross-entropy loss
              </li>
              <li>
                <strong>Backward Propagation:</strong> Compute gradients using the chain rule
              </li>
              <li>
                <strong>Parameter Updates:</strong> Update weights and biases using gradient descent
              </li>
              <li>
                <strong>Training Loop:</strong> Iterate through epochs to train the network
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <NetworkTrainingAnimation />
              </div>
            </div>
            <p>
              The interactive visualization above shows the training process in action. You can see how the loss
              decreases over time as the network learns to solve the problem.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="evaluate" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Evaluating the Neural Network</h2>
            <p className="mb-4">
              After training our neural network, we need to evaluate its performance. We'll implement methods to make
              predictions, calculate evaluation metrics, and visualize the decision boundary.
            </p>
            <p className="mb-4">Key components we'll implement:</p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Prediction:</strong> Convert network outputs to binary predictions
              </li>
              <li>
                <strong>Evaluation Metrics:</strong> Calculate accuracy, precision, recall, and F1 score
              </li>
              <li>
                <strong>Visualization:</strong> Plot the decision boundary and loss curve
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <NetworkEvaluationAnimation />
              </div>
            </div>
            <p>
              The interactive visualization above demonstrates how to evaluate a neural network's performance. You can
              see the decision boundary and how well the network classifies different inputs.
            </p>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="mt-8 space-y-6">
        <h3 className="text-xl font-semibold">Sample Code</h3>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">Step 1: Building the Network</h4>
          <CodeDownloader code={buildCode} filename="neural_network_build.py" language="python" />
        </div>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">Step 2: Training the Network</h4>
          <CodeDownloader code={trainCode} filename="neural_network_train.py" language="python" />
        </div>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">Step 3: Evaluating the Network</h4>
          <CodeDownloader code={evaluateCode} filename="neural_network_evaluate.py" language="python" />
        </div>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">Complete Neural Network Implementation</h4>
          <CodeDownloader code={fullCode} filename="neural_network_complete.py" language="python" />
        </div>
      </div>
    </div>
  )
}

