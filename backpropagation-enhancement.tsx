"use client"

import { useEffect, useRef, useState } from "react"

export default function BackpropagationEnhancement() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [learningRate, setLearningRate] = useState(0.1)
  const [autoPlay, setAutoPlay] = useState(false)

  const steps = ["Forward Pass", "Calculate Error", "Compute Gradients", "Update Weights", "Repeat"]

  useEffect(() => {
    // This function would be added to backpropagation_function.js
    const enhanceBackpropagation = () => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      // Network parameters
      const layers = [2, 3, 1] // Input, hidden, output
      let weights = [
        // Layer 1 -> 2
        [
          [0.15, 0.25], // Neuron 1 in layer 2
          [0.2, 0.3], // Neuron 2 in layer 2
          [0.35, 0.35], // Neuron 3 in layer 2
        ],
        // Layer 2 -> 3
        [
          [0.4, 0.45, 0.5], // Neuron 1 in layer 3
        ],
      ]

      let biases = [
        [0.1, 0.1, 0.1], // Layer 2
        [0.1], // Layer 3
      ]

      // Training data
      const inputs = [0.05, 0.1]
      const target = [0.01]

      // Activation function and its derivative
      const sigmoid = (x) => 1 / (1 + Math.exp(-x))
      const sigmoidDerivative = (x) => sigmoid(x) * (1 - sigmoid(x))

      // Forward pass
      const forwardPass = () => {
        const activations = [inputs]
        const zValues = []

        // For each layer (except input)
        for (let l = 0; l < weights.length; l++) {
          const layerActivations = []
          const layerZValues = []

          // For each neuron in this layer
          for (let n = 0; n < weights[l].length; n++) {
            let z = biases[l][n]

            // For each connection to the previous layer
            for (let p = 0; p < weights[l][n].length; p++) {
              z += weights[l][n][p] * activations[l][p]
            }

            layerZValues.push(z)
            layerActivations.push(sigmoid(z))
          }

          zValues.push(layerZValues)
          activations.push(layerActivations)
        }

        return { activations, zValues }
      }

      // Calculate error
      const calculateError = (output, target) => {
        let error = 0
        for (let i = 0; i < output.length; i++) {
          error += 0.5 * Math.pow(target[i] - output[i], 2)
        }
        return error
      }

      // Backpropagation
      const backpropagate = (activations, zValues) => {
        const deltas = []

        // Output layer delta
        const outputDeltas = []
        for (let i = 0; i < activations[activations.length - 1].length; i++) {
          const output = activations[activations.length - 1][i]
          outputDeltas.push((output - target[i]) * sigmoidDerivative(zValues[zValues.length - 1][i]))
        }
        deltas.unshift(outputDeltas)

        // Hidden layers deltas
        for (let l = weights.length - 2; l >= 0; l--) {
          const layerDeltas = []

          for (let n = 0; n < activations[l + 1].length; n++) {
            let delta = 0

            for (let k = 0; k < deltas[0].length; k++) {
              delta += weights[l + 1][k][n] * deltas[0][k]
            }

            delta *= sigmoidDerivative(zValues[l][n])
            layerDeltas.push(delta)
          }

          deltas.unshift(layerDeltas)
        }

        return deltas
      }

      // Update weights
      const updateWeights = (activations, deltas, learningRate) => {
        const newWeights = JSON.parse(JSON.stringify(weights))
        const newBiases = JSON.parse(JSON.stringify(biases))

        for (let l = 0; l < weights.length; l++) {
          for (let n = 0; n < weights[l].length; n++) {
            // Update bias
            newBiases[l][n] -= learningRate * deltas[l][n]

            // Update weights
            for (let p = 0; p < weights[l][n].length; p++) {
              newWeights[l][n][p] -= learningRate * deltas[l][n] * activations[l][p]
            }
          }
        }

        return { newWeights, newBiases }
      }

      // Draw the network
      const drawNetwork = (step, activations, zValues, deltas, error) => {
        const width = canvas.width
        const height = canvas.height

        // Clear canvas
        ctx.clearRect(0, 0, width, height)

        // Calculate positions
        const layerGap = width / (layers.length + 1)

        // Draw connections
        for (let l = 0; l < layers.length - 1; l++) {
          const x1 = (l + 1) * layerGap
          const x2 = (l + 2) * layerGap

          const layer1Neurons = layers[l]
          const layer2Neurons = layers[l + 1]

          const neuronGap1 = height / (layer1Neurons + 1)
          const neuronGap2 = height / (layer2Neurons + 1)

          for (let n1 = 0; n1 < layer1Neurons; n1++) {
            const y1 = (n1 + 1) * neuronGap1

            for (let n2 = 0; n2 < layer2Neurons; n2++) {
              const y2 = (n2 + 1) * neuronGap2

              // Determine connection color and width based on step
              let lineWidth = 1
              let strokeStyle = "#ccc"

              if (step >= 3) {
                // Update Weights
                const weightChange = learningRate * deltas[l][n2] * activations[l][n1]
                const absChange = Math.abs(weightChange)

                if (weightChange > 0) {
                  strokeStyle = "rgba(239, 68, 68, " + Math.min(1, absChange * 10) + ")" // Red for decrease
                } else {
                  strokeStyle = "rgba(34, 197, 94, " + Math.min(1, absChange * 10) + ")" // Green for increase
                }

                lineWidth = 1 + Math.min(3, absChange * 20)
              } else if (step >= 2) {
                // Compute Gradients
                const gradient = deltas[l][n2] * activations[l][n1]
                const absGradient = Math.abs(gradient)

                strokeStyle = "rgba(59, 130, 246, " + Math.min(1, absGradient * 10) + ")" // Blue
                lineWidth = 1 + Math.min(3, absGradient * 20)
              } else if (step >= 1) {
                // Calculate Error
                strokeStyle = "rgba(100, 116, 139, 0.5)"
              }

              // Draw connection
              ctx.beginPath()
              ctx.moveTo(x1, y1)
              ctx.lineTo(x2, y2)
              ctx.strokeStyle = strokeStyle
              ctx.lineWidth = lineWidth
              ctx.stroke()

              // Draw weight value
              if (activations) {
                const midX = (x1 + x2) / 2
                const midY = (y1 + y2) / 2

                ctx.fillStyle = "black"
                ctx.font = "10px Arial"
                ctx.fillText(weights[l][n2][n1].toFixed(2), midX, midY)
              }
            }
          }
        }

        // Draw neurons
        for (let l = 0; l < layers.length; l++) {
          const x = (l + 1) * layerGap
          const layerNeurons = layers[l]
          const neuronGap = height / (layerNeurons + 1)

          for (let n = 0; n < layerNeurons; n++) {
            const y = (n + 1) * neuronGap

            // Determine neuron color based on step and activation
            let fillStyle = "#60a5fa" // Default blue

            if (activations) {
              if (l === 0) {
                fillStyle = "#4ade80" // Input layer (green)
              } else if (l === layers.length - 1) {
                fillStyle = "#f87171" // Output layer (red)
              } else {
                // Color based on activation value
                const activation = activations[l][n]
                const intensity = Math.min(255, Math.round(activation * 255))
                fillStyle = `rgb(${intensity}, ${intensity}, 255)`
              }
            }

            // Draw neuron
            ctx.beginPath()
            ctx.arc(x, y, 15, 0, 2 * Math.PI)
            ctx.fillStyle = fillStyle
            ctx.fill()
            ctx.strokeStyle = "#333"
            ctx.lineWidth = 1
            ctx.stroke()

            // Draw activation value
            if (activations && l > 0) {
              ctx.fillStyle = "white"
              ctx.font = "10px Arial"
              ctx.textAlign = "center"
              ctx.fillText(activations[l][n].toFixed(2), x, y + 3)
            } else if (l === 0) {
              ctx.fillStyle = "white"
              ctx.font = "10px Arial"
              ctx.textAlign = "center"
              ctx.fillText(inputs[n].toFixed(2), x, y + 3)
            }

            // Draw delta value for backpropagation
            if (step >= 2 && deltas && l > 0) {
              ctx.fillStyle = "black"
              ctx.font = "10px Arial"
              ctx.textAlign = "center"
              ctx.fillText("δ: " + deltas[l - 1][n].toFixed(3), x, y - 20)
            }
          }
        }

        // Draw step information
        ctx.fillStyle = "black"
        ctx.font = "14px Arial"
        ctx.textAlign = "left"
        ctx.fillText(`Step: ${steps[step]}`, 20, 30)

        if (error !== undefined) {
          ctx.fillText(`Error: ${error.toFixed(6)}`, 20, 50)
        }

        if (step === 0) {
          ctx.fillText("Forward Pass: Calculate activations through the network", 20, height - 20)
        } else if (step === 1) {
          ctx.fillText("Calculate Error: Compare output with target", 20, height - 20)
        } else if (step === 2) {
          ctx.fillText("Compute Gradients: Calculate how each weight affects the error", 20, height - 20)
        } else if (step === 3) {
          ctx.fillText(`Update Weights: Adjust weights by learning rate (${learningRate})`, 20, height - 20)
        } else if (step === 4) {
          ctx.fillText("Repeat the process for the next training example", 20, height - 20)
        }
      }

      // Perform one step of backpropagation
      const performStep = (step) => {
        const { activations, zValues } = forwardPass()
        const error = calculateError(activations[activations.length - 1], target)
        const deltas = backpropagate(activations, zValues)

        if (step === 0) {
          // Forward Pass
          drawNetwork(step, activations, zValues, null, error)
        } else if (step === 1) {
          // Calculate Error
          drawNetwork(step, activations, zValues, null, error)
        } else if (step === 2) {
          // Compute Gradients
          drawNetwork(step, activations, zValues, deltas, error)
        } else if (step === 3) {
          // Update Weights
          drawNetwork(step, activations, zValues, deltas, error)
          const { newWeights, newBiases } = updateWeights(activations, deltas, learningRate)
          weights = newWeights
          biases = newBiases
        } else if (step === 4) {
          // Repeat
          drawNetwork(step, activations, zValues, deltas, error)
        }
      }

      // Initial draw
      performStep(currentStep)

      // Add event listeners for controls
      document.getElementById("prev-step")?.addEventListener("click", () => {
        setCurrentStep(Math.max(0, currentStep - 1))
      })

      document.getElementById("next-step")?.addEventListener("click", () => {
        setCurrentStep(Math.min(steps.length - 1, currentStep + 1))
      })

      document.getElementById("learning-rate-slider")?.addEventListener("input", (e) => {
        const newRate = Number.parseFloat((e.target as HTMLInputElement).value)
        setLearningRate(newRate)
        document.getElementById("learning-rate-value").textContent = newRate.toFixed(2)
      })

      document.getElementById("auto-play")?.addEventListener("click", () => {
        setAutoPlay(!autoPlay)
      })

      // Update when step changes
      performStep(currentStep)
    }

    enhanceBackpropagation()

    // Auto-play functionality
    let autoPlayInterval
    if (autoPlay) {
      autoPlayInterval = setInterval(() => {
        setCurrentStep((prev) => {
          const next = (prev + 1) % steps.length
          return next
        })
      }, 2000)
    }

    return () => {
      clearInterval(autoPlayInterval)
    }
  }, [currentStep, learningRate, autoPlay])

  return (
    <div className="p-4 border rounded-lg bg-white">
      <h2 className="text-xl font-semibold mb-4">Enhanced Backpropagation Visualization</h2>

      <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Current Step: {steps[currentStep]}</label>
          <div className="flex items-center space-x-2">
            <button
              id="prev-step"
              className="px-3 py-1 bg-gray-200 rounded"
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
              disabled={currentStep === 0}
            >
              Previous
            </button>
            <div className="flex-1 h-2 bg-gray-200 rounded-full">
              <div
                className="h-2 bg-blue-500 rounded-full"
                style={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
              ></div>
            </div>
            <button
              id="next-step"
              className="px-3 py-1 bg-gray-200 rounded"
              onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
              disabled={currentStep === steps.length - 1}
            >
              Next
            </button>
          </div>
        </div>

        <div>
          <label htmlFor="learning-rate-slider" className="block text-sm font-medium mb-1">
            Learning Rate
          </label>
          <input
            id="learning-rate-slider"
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={learningRate}
            onChange={(e) => setLearningRate(Number.parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-sm text-gray-600 mt-1">
            <span>0.01</span>
            <span id="learning-rate-value">{learningRate.toFixed(2)}</span>
            <span>0.50</span>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <button
          id="auto-play"
          className={`px-4 py-2 ${autoPlay ? "bg-red-500" : "bg-blue-500"} text-white rounded hover:${autoPlay ? "bg-red-600" : "bg-blue-600"}`}
          onClick={() => setAutoPlay(!autoPlay)}
        >
          {autoPlay ? "Stop Auto-Play" : "Start Auto-Play"}
        </button>
      </div>

      <div className="border p-2 bg-gray-50 rounded">
        <canvas ref={canvasRef} width="600" height="300" className="w-full h-auto"></canvas>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          This visualization shows the backpropagation algorithm step by step. Use the controls to navigate through each
          step and see how the network learns.
        </p>
        <ul className="list-disc pl-5 mt-2">
          <li>
            <strong>Forward Pass:</strong> Calculate activations through the network
          </li>
          <li>
            <strong>Calculate Error:</strong> Compare output with target
          </li>
          <li>
            <strong>Compute Gradients:</strong> Calculate how each weight affects the error
          </li>
          <li>
            <strong>Update Weights:</strong> Adjust weights by learning rate
          </li>
          <li>
            <strong>Repeat:</strong> Continue the process for more training examples
          </li>
        </ul>
      </div>
    </div>
  )
}

