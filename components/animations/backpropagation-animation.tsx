"use client"

import { useEffect, useRef, useState } from "react"

export default function BackpropagationAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [step, setStep] = useState(0)
  const [learningRate, setLearningRate] = useState(0.1)
  const [autoPlay, setAutoPlay] = useState(false)
  const [animationFrame, setAnimationFrame] = useState<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    const resizeCanvas = () => {
      const parent = canvas.parentElement
      if (parent) {
        canvas.width = parent.clientWidth
        canvas.height = parent.clientHeight
      }
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

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

    // Neuron positions
    const neurons: { x: number; y: number; layer: number; index: number }[] = []

    // Initialize neurons
    const initializeNetwork = () => {
      neurons.length = 0

      const padding = 50
      const layerSpacing = (canvas.width - padding * 2) / (layers.length - 1)

      // Create neurons
      let neuronIndex = 0
      for (let l = 0; l < layers.length; l++) {
        const neuronsInLayer = layers[l]
        const layerHeight = neuronsInLayer * 60
        const startY = (canvas.height - layerHeight) / 2

        for (let n = 0; n < neuronsInLayer; n++) {
          neurons.push({
            x: padding + l * layerSpacing,
            y: startY + n * 60 + 30,
            layer: l,
            index: neuronIndex++,
          })
        }
      }
    }

    // Activation function and its derivative
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
    const sigmoidDerivative = (x: number) => sigmoid(x) * (1 - sigmoid(x))

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
    const calculateError = (output: number[], target: number[]) => {
      let error = 0
      for (let i = 0; i < output.length; i++) {
        error += 0.5 * Math.pow(target[i] - output[i], 2)
      }
      return error
    }

    // Backpropagation
    const backpropagate = (activations: number[][], zValues: number[][]) => {
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
    const updateWeights = (activations: number[][], deltas: number[][], learningRate: number) => {
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
    const drawNetwork = (step: number) => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Perform forward pass
      const { activations, zValues } = forwardPass()
      const error = calculateError(activations[activations.length - 1], target)
      const deltas = backpropagate(activations, zValues)

      // Draw connections
      for (let l = 0; l < layers.length - 1; l++) {
        const startNeurons = neurons.filter((n) => n.layer === l)
        const endNeurons = neurons.filter((n) => n.layer === l + 1)

        for (const start of startNeurons) {
          for (const end of endNeurons) {
            const startIdx = start.index - (l === 0 ? 0 : layers[0])
            const endIdx = end.index - (layers[0] + (l === 0 ? 0 : layers[1]))

            // Determine connection color and width based on step
            let lineWidth = 1
            let strokeStyle = "#cbd5e1"

            if (step >= 3) {
              // Update Weights
              const weightChange = learningRate * deltas[l][endIdx] * activations[l][startIdx]
              const absChange = Math.abs(weightChange)

              if (weightChange > 0) {
                strokeStyle = `rgba(239, 68, 68, ${Math.min(1, absChange * 10)})` // Red for decrease
              } else {
                strokeStyle = `rgba(34, 197, 94, ${Math.min(1, absChange * 10)})` // Green for increase
              }

              lineWidth = 1 + Math.min(3, absChange * 20)
            } else if (step >= 2) {
              // Compute Gradients
              const gradient = deltas[l][endIdx] * activations[l][startIdx]
              const absGradient = Math.abs(gradient)

              strokeStyle = `rgba(59, 130, 246, ${Math.min(1, absGradient * 10)})` // Blue
              lineWidth = 1 + Math.min(3, absGradient * 20)
            }

            // Draw connection
            ctx.beginPath()
            ctx.moveTo(start.x, start.y)
            ctx.lineTo(end.x, end.y)
            ctx.strokeStyle = strokeStyle
            ctx.lineWidth = lineWidth
            ctx.stroke()

            // Draw weight value
            const midX = (start.x + end.x) / 2
            const midY = (start.y + end.y) / 2

            ctx.fillStyle = "#1e293b"
            ctx.font = "12px Arial"
            ctx.textAlign = "center"
            ctx.textBaseline = "middle"

            const weightValue = l === 0 ? weights[l][endIdx][startIdx] : weights[l][0][endIdx]

            ctx.fillText(weightValue.toFixed(2), midX, midY)
          }
        }
      }

      // Draw neurons
      for (const neuron of neurons) {
        // Determine neuron color based on layer
        let fillStyle
        if (neuron.layer === 0) {
          fillStyle = "#10b981" // Input layer (green)
        } else if (neuron.layer === layers.length - 1) {
          fillStyle = "#ef4444" // Output layer (red)
        } else {
          fillStyle = "#3b82f6" // Hidden layer (blue)
        }

        // Draw neuron
        ctx.beginPath()
        ctx.arc(neuron.x, neuron.y, 20, 0, Math.PI * 2)
        ctx.fillStyle = fillStyle
        ctx.fill()
        ctx.strokeStyle = "#1e293b"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw neuron value
        ctx.fillStyle = "#ffffff"
        ctx.font = "12px Arial"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"

        let value
        if (neuron.layer === 0) {
          value = inputs[neuron.index]
        } else if (neuron.layer === 1) {
          value = activations[1][neuron.index - layers[0]]
        } else {
          value = activations[2][0]
        }

        ctx.fillText(value.toFixed(2), neuron.x, neuron.y)

        // Draw delta value for backpropagation
        if (step >= 2 && neuron.layer > 0) {
          let deltaValue
          if (neuron.layer === 1) {
            deltaValue = deltas[0][neuron.index - layers[0]]
          } else {
            deltaValue = deltas[1][0]
          }

          ctx.fillStyle = "#1e293b"
          ctx.font = "12px Arial"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(`δ: ${deltaValue.toFixed(3)}`, neuron.x, neuron.y - 30)
        }
      }

      // Draw step information
      ctx.fillStyle = "#1e293b"
      ctx.font = "16px Arial"
      ctx.textAlign = "left"
      ctx.textBaseline = "top"

      const stepNames = ["Forward Pass", "Calculate Error", "Compute Gradients", "Update Weights", "Repeat"]

      ctx.fillText(`Step: ${stepNames[step]}`, 20, 20)
      ctx.fillText(`Error: ${error.toFixed(6)}`, 20, 45)
      ctx.fillText(`Learning Rate: ${learningRate}`, 20, 70)

      // Draw step-specific information
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "bottom"

      let stepInfo
      switch (step) {
        case 0:
          stepInfo = "Forward Pass: Calculate activations through the network"
          break
        case 1:
          stepInfo = `Calculate Error: Target = ${target[0]}, Output = ${activations[2][0].toFixed(4)}`
          break
        case 2:
          stepInfo = "Compute Gradients: Calculate how each weight affects the error"
          break
        case 3:
          stepInfo = "Update Weights: Adjust weights to reduce error"
          break
        case 4:
          stepInfo = "Repeat the process for the next training example"
          break
        default:
          stepInfo = ""
      }

      ctx.fillText(stepInfo, canvas.width / 2, canvas.height - 20)

      // If we're at the update weights step, actually update the weights
      if (step === 3) {
        const { newWeights, newBiases } = updateWeights(activations, deltas, learningRate)
        weights = newWeights
        biases = newBiases
      }
    }

    // Initialize network
    initializeNetwork()

    // Draw initial state
    drawNetwork(step)

    // Set up auto-play if enabled
    let autoPlayInterval: NodeJS.Timeout | null = null

    if (autoPlay) {
      autoPlayInterval = setInterval(() => {
        setStep((prevStep) => (prevStep + 1) % 5)
      }, 2000)
    }

    // Cleanup
    return () => {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval)
      }
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [step, learningRate, autoPlay])

  // Handle step change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Redraw network with new step
    const drawNetwork = () => {
      // Implementation would be the same as in the main effect
      // This is a simplified version that just triggers a redraw
      const frame = requestAnimationFrame(() => {
        // The main effect will handle the actual drawing
      })
      setAnimationFrame(frame)
    }

    drawNetwork()
  }, [step, learningRate])

  // Handle auto-play toggle
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null

    if (autoPlay) {
      interval = setInterval(() => {
        setStep((prevStep) => (prevStep + 1) % 5)
      }, 2000)
    }

    return () => {
      if (interval) {
        clearInterval(interval)
      }
    }
  }, [autoPlay])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setStep((prevStep) => (prevStep - 1 + 5) % 5)}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300"
            disabled={autoPlay}
          >
            Previous
          </button>
          <button
            onClick={() => setStep((prevStep) => (prevStep + 1) % 5)}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300"
            disabled={autoPlay}
          >
            Next
          </button>
        </div>

        <div className="flex items-center space-x-2">
          <label>Learning Rate:</label>
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={learningRate}
            onChange={(e) => setLearningRate(Number.parseFloat(e.target.value))}
            className="w-32"
            disabled={autoPlay}
          />
          <span>{learningRate.toFixed(2)}</span>
        </div>

        <button
          onClick={() => setAutoPlay(!autoPlay)}
          className={`px-3 py-1 rounded ${
            autoPlay ? "bg-red-500 text-white hover:bg-red-600" : "bg-blue-500 text-white hover:bg-blue-600"
          }`}
        >
          {autoPlay ? "Stop Auto-Play" : "Start Auto-Play"}
        </button>
      </div>

      <canvas ref={canvasRef} className="flex-1 w-full border rounded" />
    </div>
  )
}

