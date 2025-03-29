"use client"

import { useEffect, useRef, useState } from "react"

export default function NetworkTrainingAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [animationFrame, setAnimationFrame] = useState<number | null>(null)
  const [epoch, setEpoch] = useState(0)
  const [loss, setLoss] = useState(1.0)
  const [learningRate, setLearningRate] = useState(0.1)
  const [isTraining, setIsTraining] = useState(true)

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
    const layers = [2, 4, 1]
    const neurons: { x: number; y: number; layer: number; activation: number }[] = []
    const connections: { from: number; to: number; weight: number }[] = []
    const signals: { x: number; y: number; targetX: number; targetY: number; progress: number; value: number }[] = []

    // Training data (XOR problem)
    const trainingData = [
      { inputs: [0, 0], target: 0 },
      { inputs: [0, 1], target: 1 },
      { inputs: [1, 0], target: 1 },
      { inputs: [1, 1], target: 0 },
    ]

    // Loss history
    const lossHistory: number[] = []
    const maxLossHistoryPoints = 100

    // Initialize neurons and connections
    const initializeNetwork = () => {
      neurons.length = 0
      connections.length = 0

      const padding = 50
      const networkWidth = canvas.width * 0.6 - padding * 2
      const networkHeight = canvas.height - padding * 2

      // Calculate positions
      const layerSpacing = networkWidth / (layers.length - 1)

      // Create neurons
      let neuronIndex = 0
      for (let l = 0; l < layers.length; l++) {
        const neuronsInLayer = layers[l]
        const layerHeight = neuronsInLayer * 40
        const startY = padding + (networkHeight - layerHeight) / 2

        for (let n = 0; n < neuronsInLayer; n++) {
          neurons.push({
            x: padding + l * layerSpacing,
            y: startY + n * 40 + 20,
            layer: l,
            activation: 0,
          })
          neuronIndex++
        }
      }

      // Create connections with random weights
      for (let l = 0; l < layers.length - 1; l++) {
        const startNeurons = neurons.filter((n) => n.layer === l)
        const endNeurons = neurons.filter((n) => n.layer === l + 1)

        for (const start of startNeurons) {
          for (const end of endNeurons) {
            connections.push({
              from: neurons.indexOf(start),
              to: neurons.indexOf(end),
              weight: Math.random() * 2 - 1, // Random weight between -1 and 1
            })
          }
        }
      }
    }

    // Forward pass
    const forwardPass = (inputs: number[]) => {
      // Set input layer activations
      const inputNeurons = neurons.filter((n) => n.layer === 0)
      for (let i = 0; i < inputNeurons.length; i++) {
        inputNeurons[i].activation = inputs[i]
      }

      // Clear other activations
      for (const neuron of neurons) {
        if (neuron.layer > 0) {
          neuron.activation = 0
        }
      }

      // Propagate signals
      for (let l = 0; l < layers.length - 1; l++) {
        const currentNeurons = neurons.filter((n) => n.layer === l)
        const nextNeurons = neurons.filter((n) => n.layer === l + 1)

        // For each neuron in the current layer
        for (const current of currentNeurons) {
          // Find all connections from this neuron
          const outgoingConnections = connections.filter((c) => c.from === neurons.indexOf(current))

          // For each connection
          for (const conn of outgoingConnections) {
            const targetNeuron = neurons[conn.to]

            // Add weighted input to target neuron
            targetNeuron.activation += current.activation * conn.weight

            // Create a signal
            signals.push({
              x: current.x,
              y: current.y,
              targetX: targetNeuron.x,
              targetY: targetNeuron.y,
              progress: 0,
              value: current.activation * conn.weight,
            })
          }
        }

        // Apply activation function (sigmoid) to next layer
        for (const next of nextNeurons) {
          next.activation = 1 / (1 + Math.exp(-next.activation))
        }
      }

      // Return output
      const outputNeuron = neurons.find((n) => n.layer === layers.length - 1)
      return outputNeuron ? outputNeuron.activation : 0
    }

    // Calculate loss (mean squared error)
    const calculateLoss = () => {
      let totalLoss = 0

      for (const data of trainingData) {
        const output = forwardPass(data.inputs)
        totalLoss += Math.pow(data.target - output, 2)
      }

      return totalLoss / trainingData.length
    }

    // Update weights (simplified backpropagation)
    const updateWeights = () => {
      const learningFactor = learningRate * (1 - loss) // Adaptive learning rate

      for (const conn of connections) {
        // Simplified weight update - in reality, this would use proper backpropagation
        conn.weight += (Math.random() * 2 - 1) * learningFactor
      }
    }

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update signals
      for (let i = 0; i < signals.length; i++) {
        const signal = signals[i]
        signal.progress += 0.02

        if (signal.progress >= 1) {
          signals.splice(i, 1)
          i--
        }
      }

      // Draw network
      drawNetwork()

      // Draw loss graph
      drawLossGraph()

      // Training step
      if (isTraining && epoch < 1000) {
        // Every 5 frames, perform a training step
        if (epoch % 5 === 0) {
          // Calculate current loss
          const currentLoss = calculateLoss()
          setLoss(currentLoss)

          // Add to history
          lossHistory.push(currentLoss)
          if (lossHistory.length > maxLossHistoryPoints) {
            lossHistory.shift()
          }

          // Update weights
          updateWeights()

          // Create new signals for visualization
          createSignals()
        }

        setEpoch(epoch + 1)
      }

      // Continue animation
      const frame = requestAnimationFrame(animate)
      setAnimationFrame(frame)
    }

    // Create signals for visualization
    const createSignals = () => {
      // Clear existing signals
      signals.length = 0

      // Choose a random training example
      const example = trainingData[Math.floor(Math.random() * trainingData.length)]

      // Set input layer activations
      const inputNeurons = neurons.filter((n) => n.layer === 0)
      for (let i = 0; i < inputNeurons.length; i++) {
        inputNeurons[i].activation = example.inputs[i]
      }

      // Create signals from input layer
      for (const neuron of inputNeurons) {
        const outgoingConnections = connections.filter((c) => c.from === neurons.indexOf(neuron))

        for (const conn of outgoingConnections) {
          const targetNeuron = neurons[conn.to]

          signals.push({
            x: neuron.x,
            y: neuron.y,
            targetX: targetNeuron.x,
            targetY: targetNeuron.y,
            progress: 0,
            value: neuron.activation * conn.weight,
          })
        }
      }
    }

    // Draw the network
    const drawNetwork = () => {
      // Draw connections
      for (const conn of connections) {
        const start = neurons[conn.from]
        const end = neurons[conn.to]

        // Color based on weight
        const weightColor =
          conn.weight >= 0
            ? `rgba(16, 185, 129, ${Math.min(1, Math.abs(conn.weight) + 0.2)})` // Green for positive
            : `rgba(239, 68, 68, ${Math.min(1, Math.abs(conn.weight) + 0.2)})` // Red for negative

        ctx.beginPath()
        ctx.moveTo(start.x, start.y)
        ctx.lineTo(end.x, end.y)
        ctx.strokeStyle = weightColor
        ctx.lineWidth = Math.max(1, Math.abs(conn.weight) * 3)
        ctx.stroke()
      }

      // Draw signals
      for (const signal of signals) {
        const x = signal.x + (signal.targetX - signal.x) * signal.progress
        const y = signal.y + (signal.targetY - signal.y) * signal.progress

        // Color based on value
        const signalColor =
          signal.value >= 0
            ? `rgba(16, 185, 129, ${Math.min(1, Math.abs(signal.value) + 0.2)})` // Green for positive
            : `rgba(239, 68, 68, ${Math.min(1, Math.abs(signal.value) + 0.2)})` // Red for negative

        ctx.beginPath()
        ctx.arc(x, y, 4, 0, Math.PI * 2)
        ctx.fillStyle = signalColor
        ctx.fill()
      }

      // Draw neurons
      for (const neuron of neurons) {
        ctx.beginPath()
        ctx.arc(neuron.x, neuron.y, 15, 0, Math.PI * 2)

        // Color based on layer and activation
        if (neuron.layer === 0) {
          ctx.fillStyle = `rgba(16, 185, 129, ${neuron.activation * 0.8 + 0.2})` // Input layer - green
        } else if (neuron.layer === layers.length - 1) {
          ctx.fillStyle = `rgba(239, 68, 68, ${neuron.activation * 0.8 + 0.2})` // Output layer - red
        } else {
          ctx.fillStyle = `rgba(59, 130, 246, ${neuron.activation * 0.8 + 0.2})` // Hidden layers - blue
        }

        ctx.fill()
        ctx.strokeStyle = "#1e293b"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw activation value
        ctx.fillStyle = "#ffffff"
        ctx.font = "12px Arial"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(neuron.activation.toFixed(2), neuron.x, neuron.y)
      }

      // Draw layer labels
      ctx.fillStyle = "#1e293b"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"

      const inputX = neurons.find((n) => n.layer === 0)?.x || 0
      const hiddenX = neurons.find((n) => n.layer === 1)?.x || 0
      const outputX = neurons.find((n) => n.layer === 2)?.x || 0

      ctx.fillText("Input Layer", inputX, 20)
      ctx.fillText("Hidden Layer", hiddenX, 20)
      ctx.fillText("Output Layer", outputX, 20)
    }

    // Draw loss graph
    const drawLossGraph = () => {
      const graphX = canvas.width * 0.65
      const graphY = 50
      const graphWidth = canvas.width * 0.3
      const graphHeight = canvas.height - 100

      // Draw graph background
      ctx.fillStyle = "#f8fafc"
      ctx.fillRect(graphX, graphY, graphWidth, graphHeight)

      // Draw border
      ctx.strokeStyle = "#cbd5e1"
      ctx.lineWidth = 1
      ctx.strokeRect(graphX, graphY, graphWidth, graphHeight)

      // Draw title
      ctx.fillStyle = "#1e293b"
      ctx.font = "16px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Training Loss", graphX + graphWidth / 2, 20)

      // Draw axes labels
      ctx.font = "12px Arial"
      ctx.textAlign = "center"
      ctx.fillText("Epoch", graphX + graphWidth / 2, graphY + graphHeight + 10)

      ctx.save()
      ctx.translate(graphX - 10, graphY + graphHeight / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText("Loss", 0, 0)
      ctx.restore()

      // Draw loss curve
      if (lossHistory.length > 1) {
        ctx.beginPath()

        const pointWidth = graphWidth / maxLossHistoryPoints

        for (let i = 0; i < lossHistory.length; i++) {
          const x = graphX + i * pointWidth
          const y = graphY + graphHeight - lossHistory[i] * graphHeight

          if (i === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        }

        ctx.strokeStyle = "#3b82f6"
        ctx.lineWidth = 2
        ctx.stroke()
      }

      // Draw current loss and epoch
      ctx.fillStyle = "#1e293b"
      ctx.font = "14px Arial"
      ctx.textAlign = "left"
      ctx.textBaseline = "top"
      ctx.fillText(`Current Loss: ${loss.toFixed(4)}`, graphX, graphY + graphHeight + 30)
      ctx.fillText(`Epoch: ${epoch}`, graphX + graphWidth - 100, graphY + graphHeight + 30)
    }

    // Initialize and start animation
    initializeNetwork()
    createSignals()
    const frame = requestAnimationFrame(animate)
    setAnimationFrame(frame)

    // Cleanup
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [epoch, loss, learningRate, isTraining])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex justify-between items-center mb-2">
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
          />
          <span>{learningRate.toFixed(2)}</span>
        </div>

        <button
          onClick={() => setIsTraining(!isTraining)}
          className={`px-3 py-1 rounded ${
            isTraining ? "bg-red-500 text-white hover:bg-red-600" : "bg-blue-500 text-white hover:bg-blue-600"
          }`}
        >
          {isTraining ? "Pause Training" : "Resume Training"}
        </button>

        <button
          onClick={() => {
            setEpoch(0)
            setLoss(1.0)
          }}
          className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300"
        >
          Reset
        </button>
      </div>

      <canvas ref={canvasRef} className="flex-1 w-full border rounded" />
    </div>
  )
}

