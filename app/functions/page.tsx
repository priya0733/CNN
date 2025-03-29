"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import ActivationFunctionsAnimation from "@/components/animations/activation-functions-animation"
import LossFunctionsAnimation from "@/components/animations/loss-functions-animation"
import BackpropagationAnimation from "@/components/animations/backpropagation-animation"
import CodeDownloader from "@/components/code-downloader"

export default function FunctionsPage() {
  const [activeTab, setActiveTab] = useState("activation")

  const activationCode = `
# Implementing common activation functions
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def tanh(x):
    return np.tanh(x)

# Generate data for plotting
x = np.linspace(-5, 5, 100)

# Plot activation functions
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.tight_layout()
plt.show()
`

  const lossCode = `
# Implementing common loss functions
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generate sample data
y_true = np.array([1, 0, 1, 1, 0])
predictions = np.linspace(0, 1, 100)

# Calculate loss for different prediction values
mse_values = [mean_squared_error(y_true, np.full_like(y_true, p)) for p in predictions]
mae_values = [mean_absolute_error(y_true, np.full_like(y_true, p)) for p in predictions]
bce_values = [binary_cross_entropy(y_true, np.full_like(y_true, p)) for p in predictions]

# Plot loss functions
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(predictions, mse_values)
plt.title('Mean Squared Error')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(predictions, mae_values)
plt.title('Mean Absolute Error')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(predictions, bce_values)
plt.title('Binary Cross Entropy')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
`

  const backpropCode = `
# Simple backpropagation implementation
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        m = X.shape[0]
        
        # Calculate gradients
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = np.array([[0], [1], [1], [0]])              # XOR outputs

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Test the trained network
predictions = nn.forward(X)
print("Predictions after training:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {predictions[i]}")
`

  return (
    <div className="container mx-auto space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Activation Functions, Loss Functions, and Backpropagation</h1>
        <p className="text-muted-foreground">Understand the mathematical building blocks of neural networks</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="activation">Activation Functions</TabsTrigger>
          <TabsTrigger value="loss">Loss Functions</TabsTrigger>
          <TabsTrigger value="backprop">Backpropagation</TabsTrigger>
        </TabsList>

        <TabsContent value="activation" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Activation Functions</h2>
            <p className="mb-4">
              Activation functions introduce non-linearity into neural networks, allowing them to learn complex
              patterns. Without activation functions, a neural network would behave like a linear regression model,
              regardless of its depth.
            </p>
            <p className="mb-4">Common activation functions include:</p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Sigmoid:</strong> Maps values to range (0,1), useful for output layers in binary classification
              </li>
              <li>
                <strong>ReLU (Rectified Linear Unit):</strong> Returns x if x &gt; 0, else 0; most commonly used in
                hidden layers
              </li>
              <li>
                <strong>Leaky ReLU:</strong> Returns x if x &gt; 0, else αx (where α is small); helps with "dying ReLU"
                problem
              </li>
              <li>
                <strong>Tanh:</strong> Maps values to range (-1,1), similar to sigmoid but zero-centered
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <ActivationFunctionsAnimation />
              </div>
            </div>
            <p>
              The interactive visualization above allows you to explore different activation functions and see how they
              transform input values. Try adjusting the parameters to see how they affect the function's behavior.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="loss" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Loss Functions</h2>
            <p className="mb-4">
              Loss functions measure how well a neural network performs by quantifying the difference between predicted
              outputs and actual target values. The goal of training is to minimize this loss.
            </p>
            <p className="mb-4">Common loss functions include:</p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Mean Squared Error (MSE):</strong> Average of squared differences; used for regression tasks
              </li>
              <li>
                <strong>Mean Absolute Error (MAE):</strong> Average of absolute differences; less sensitive to outliers
              </li>
              <li>
                <strong>Binary Cross-Entropy:</strong> Measures performance for binary classification tasks
              </li>
              <li>
                <strong>Categorical Cross-Entropy:</strong> Used for multi-class classification problems
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <LossFunctionsAnimation />
              </div>
            </div>
            <p>
              The interactive visualization above demonstrates how different loss functions behave as the predicted
              values approach or diverge from the target values. Experiment with different scenarios to understand when
              each loss function is most appropriate.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="backprop" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Backpropagation</h2>
            <p className="mb-4">
              Backpropagation is the algorithm used to train neural networks by adjusting the weights and biases to
              minimize the loss function. It works by calculating gradients of the loss function with respect to each
              weight using the chain rule of calculus.
            </p>
            <p className="mb-4">The backpropagation process involves:</p>
            <ol className="list-decimal pl-6 mb-4 space-y-2">
              <li>
                <strong>Forward Pass:</strong> Compute the output of the network given the inputs
              </li>
              <li>
                <strong>Compute Loss:</strong> Calculate the error between predicted and actual outputs
              </li>
              <li>
                <strong>Backward Pass:</strong> Propagate the error backwards through the network
              </li>
              <li>
                <strong>Update Weights:</strong> Adjust weights and biases using gradient descent
              </li>
            </ol>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <BackpropagationAnimation />
              </div>
            </div>
            <p>
              The interactive visualization above illustrates the backpropagation process step by step. You can see how
              errors propagate backward through the network and how weights are updated to improve the network's
              performance.
            </p>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="mt-8 space-y-6">
        <h3 className="text-xl font-semibold">Sample Code</h3>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">Activation Functions Implementation</h4>
          <CodeDownloader code={activationCode} filename="activation_functions.py" language="python" />
        </div>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">Loss Functions Implementation</h4>
          <CodeDownloader code={lossCode} filename="loss_functions.py" language="python" />
        </div>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">Backpropagation Implementation</h4>
          <CodeDownloader code={backpropCode} filename="backpropagation.py" language="python" />
        </div>
      </div>
    </div>
  )
}

