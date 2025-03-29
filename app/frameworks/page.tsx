"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import TensorFlowAnimation from "@/components/animations/tensorflow-animation"
import PyTorchAnimation from "@/components/animations/pytorch-animation"
import FrameworkComparison from "@/components/animations/framework-comparison"
import CodeDownloader from "@/components/code-downloader"

export default function FrameworksPage() {
  const [activeTab, setActiveTab] = useState("tensorflow")

  const tensorflowCode = `
# TensorFlow example: Simple neural network
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Create some sample data
X_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
X_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

# Build a simple model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
`

  const pytorchCode = `
# PyTorch example: Simple neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create some sample data
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 2, (1000, 1)).float()
X_test = torch.randn(100, 20)
y_test = torch.randint(0, 2, (100, 1)).float()

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(20, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.output(x))
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")
`

  return (
    <div className="container mx-auto space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">TensorFlow & PyTorch Basics</h1>
        <p className="text-muted-foreground">Learn the fundamentals of the most popular deep learning frameworks</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="tensorflow">TensorFlow</TabsTrigger>
          <TabsTrigger value="pytorch">PyTorch</TabsTrigger>
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
        </TabsList>

        <TabsContent value="tensorflow" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">TensorFlow</h2>
            <p className="mb-4">
              TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive
              ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in
              ML and developers easily build and deploy ML-powered applications.
            </p>
            <p className="mb-4">Key features of TensorFlow:</p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Keras Integration:</strong> High-level API for building and training models
              </li>
              <li>
                <strong>TensorFlow.js:</strong> Machine learning in JavaScript
              </li>
              <li>
                <strong>TensorFlow Lite:</strong> Deployment on mobile and IoT devices
              </li>
              <li>
                <strong>TensorFlow Extended (TFX):</strong> End-to-end platform for deploying production ML pipelines
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <TensorFlowAnimation />
              </div>
            </div>
            <p>
              The animation above demonstrates the computational graph approach used by TensorFlow, showing how data
              flows through operations in a neural network.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="pytorch" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">PyTorch</h2>
            <p className="mb-4">
              PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It's known for
              its flexibility and dynamic computational graph, making it popular among researchers and those working on
              complex models.
            </p>
            <p className="mb-4">Key features of PyTorch:</p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Dynamic Computational Graph:</strong> Define-by-run approach for more intuitive debugging
              </li>
              <li>
                <strong>TorchScript:</strong> Seamless transition between eager mode and graph mode
              </li>
              <li>
                <strong>Distributed Training:</strong> Built-in support for distributed training
              </li>
              <li>
                <strong>Ecosystem:</strong> Rich ecosystem of tools and libraries (TorchVision, TorchText, etc.)
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <PyTorchAnimation />
              </div>
            </div>
            <p>
              The animation above illustrates PyTorch's dynamic computational graph, showing how operations are executed
              as they're defined, which is different from TensorFlow's static graph approach.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">TensorFlow vs PyTorch</h2>
            <p className="mb-4">
              Both TensorFlow and PyTorch are powerful frameworks with their own strengths and weaknesses. The choice
              between them often depends on your specific use case and preferences.
            </p>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <FrameworkComparison />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">TensorFlow Advantages</h3>
                <ul className="list-disc pl-6 space-y-1">
                  <li>Production-ready deployment options</li>
                  <li>TensorBoard for visualization</li>
                  <li>Mobile and edge deployment (TF Lite)</li>
                  <li>Larger community and more tutorials</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2">PyTorch Advantages</h3>
                <ul className="list-disc pl-6 space-y-1">
                  <li>More pythonic and intuitive API</li>
                  <li>Dynamic computational graph</li>
                  <li>Easier debugging</li>
                  <li>More popular in research community</li>
                </ul>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="mt-8 space-y-6">
        <h3 className="text-xl font-semibold">Sample Code</h3>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">TensorFlow Example</h4>
          <CodeDownloader code={tensorflowCode} filename="tensorflow_example.py" language="python" />
        </div>

        <div className="space-y-4">
          <h4 className="text-lg font-medium">PyTorch Example</h4>
          <CodeDownloader code={pytorchCode} filename="pytorch_example.py" language="python" />
        </div>
      </div>
    </div>
  )
}

