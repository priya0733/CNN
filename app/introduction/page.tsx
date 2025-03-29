"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import NeuronAnimation from "@/components/animations/neuron-animation"
import NetworkLayersAnimation from "@/components/animations/network-layers-animation"
import DeepLearningApplications from "@/components/animations/deep-learning-applications"
import CodeDownloader from "@/components/code-downloader"

export default function IntroductionPage() {
  const [activeTab, setActiveTab] = useState("what-is-dl")

  const pythonCode = `
# Simple neuron implementation
import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward(self, inputs):
        # Weight inputs, add bias, then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(total)

# Example usage
weights = np.array([0.5, -0.5, 0.3])  # Example weights
bias = 0.1                            # Example bias
neuron = Neuron(weights, bias)

# Calculate output for sample input
inputs = np.array([0.2, 0.3, 0.5])    # Example inputs
output = neuron.forward(inputs)
print(f"Neuron output: {output}")
`

  return (
    <div className="container mx-auto space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Introduction to Deep Learning and Neural Networks</h1>
        <p className="text-muted-foreground">
          Understand the fundamentals of neural networks and how deep learning works
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="what-is-dl">What is Deep Learning?</TabsTrigger>
          <TabsTrigger value="neural-networks">Neural Networks</TabsTrigger>
          <TabsTrigger value="applications">Applications</TabsTrigger>
        </TabsList>

        <TabsContent value="what-is-dl" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">What is Deep Learning?</h2>
            <p className="mb-4">
              Deep Learning is a subset of machine learning that uses neural networks with multiple layers (deep neural
              networks) to analyze various factors of data. Unlike traditional machine learning algorithms, deep
              learning can automatically discover the representations needed for feature detection or classification
              from raw data.
            </p>
            <p className="mb-4">
              The "deep" in deep learning refers to the number of layers through which the data is transformed. Each
              layer extracts features at different levels of abstraction, allowing the network to learn complex patterns
              in the data.
            </p>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <NeuronAnimation />
              </div>
            </div>
            <p>
              The animation above shows how a single neuron processes information, which is the fundamental building
              block of neural networks and deep learning systems.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="neural-networks" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Neural Networks</h2>
            <p className="mb-4">
              Neural networks are computing systems inspired by the biological neural networks in animal brains. They
              consist of artificial neurons organized in layers:
            </p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Input Layer:</strong> Receives the initial data
              </li>
              <li>
                <strong>Hidden Layers:</strong> Process the data through weighted connections
              </li>
              <li>
                <strong>Output Layer:</strong> Produces the final result
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <NetworkLayersAnimation />
              </div>
            </div>
            <p>
              The animation above demonstrates how information flows through a neural network, from the input layer
              through hidden layers to the output layer. Each connection has a weight that adjusts during training to
              improve the network's accuracy.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="applications" className="space-y-4">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">Applications of Deep Learning</h2>
            <p className="mb-4">
              Deep learning has revolutionized many fields and is being applied to solve complex problems:
            </p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Computer Vision:</strong> Image recognition, object detection, facial recognition
              </li>
              <li>
                <strong>Natural Language Processing:</strong> Translation, sentiment analysis, chatbots
              </li>
              <li>
                <strong>Speech Recognition:</strong> Voice assistants, transcription services
              </li>
              <li>
                <strong>Healthcare:</strong> Disease diagnosis, drug discovery, medical image analysis
              </li>
              <li>
                <strong>Autonomous Vehicles:</strong> Self-driving cars, drones
              </li>
            </ul>
            <div className="flex justify-center my-8">
              <div className="w-full max-w-2xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                <DeepLearningApplications />
              </div>
            </div>
            <p>The visualization above showcases various applications of deep learning across different domains.</p>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-4">Sample Code: Simple Neuron Implementation</h3>
        <CodeDownloader code={pythonCode} filename="simple_neuron.py" language="python" />
      </div>
    </div>
  )
}

