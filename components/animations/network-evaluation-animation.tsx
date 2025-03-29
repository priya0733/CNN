"use client"

import { useEffect, useRef, useState } from "react"

export default function NetworkEvaluationAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [animationFrame, setAnimationFrame] = useState<number | null>(null)
  const [testPoint, setTestPoint] = useState<[number, number]>([0.5, 0.5])
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(true)

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

    // Network parameters - pretrained for XOR problem
    const weights = [
      // Layer 1 -> 2
      [
        [2.9, -2.9], // Neuron 1 in hidden layer
        [-2.9, 2.9], // Neuron 2 in hidden layer
        [1.0, 1.0], // Neuron 3 in hidden layer
        [-1.0, -1.0], // Neuron 4 in hidden layer
      ],
      // Layer 2 -> 3
      [
        [2.9, 2.9, -1.5, -1.5], // Output neuron
      ],
    ]

    const biases = [
      [-1.0, 1.0, -0.5, 0.5], // Hidden layer
      [-1.0], // Output layer
    ]

    // Sigmoid activation function
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

    // Forward pass
    const predict = (x: number, y: number) => {
      // Input layer
      const inputs = [x, y]

      // Hidden layer
      const hidden = []
      for (let i = 0; i < weights[0].length; i++) {
        let sum = biases[0][i]
        for (let j = 0; j < inputs.length; j++) {
          sum += inputs[j] * weights[0][i][j]
        }
        hidden.push(sigmoid(sum))
      }

      // Output layer
      let output = biases[1][0]
      for (let i = 0; i < hidden.length; i++) {
        output += hidden[i] * weights[1][0][i]
      }

      return sigmoid(output)
    }

    // Draw function
    const draw = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const padding = 50
      const plotSize = Math.min(canvas.width, canvas.height) - padding * 2

      // Draw decision boundary
      if (showDecisionBoundary) {
        const resolution = 50
        const stepSize = plotSize / resolution

        for (let i = 0; i < resolution; i++) {
          for (let j = 0; j < resolution; j++) {
            const x = i / resolution
            const y = j / resolution

            const output = predict(x, y)

            ctx.fillStyle = `rgba(${output < 0.5 ? "16, 185, 129" : "239, 68, 68"}, 0.2)`
            ctx.fillRect(padding + i * stepSize, padding + j * stepSize, stepSize, stepSize)
          }
        }
      }

      // Draw grid
      ctx.strokeStyle = "#cbd5e1"
      ctx.lineWidth = 1

      // Vertical grid lines
      for (let i = 0; i <= 10; i++) {
        const x = padding + (i / 10) * plotSize
        ctx.beginPath()
        ctx.moveTo(x, padding)
        ctx.lineTo(x, padding + plotSize)
        ctx.stroke()
      }

      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = padding + (i / 10) * plotSize
        ctx.beginPath()
        ctx.moveTo(padding, y)
        ctx.lineTo(padding + plotSize, y)
        ctx.stroke()
      }

      // Draw axes
      ctx.strokeStyle = "#1e293b"
      ctx.lineWidth = 2

      // X-axis
      ctx.beginPath()
      ctx.moveTo(padding, padding + plotSize)
      ctx.lineTo(padding + plotSize, padding + plotSize)
      ctx.stroke()

      // Y-axis
      ctx.beginPath()
      ctx.moveTo(padding, padding)
      ctx.lineTo(padding, padding + plotSize)
      ctx.stroke()

      // Draw axis labels
      ctx.fillStyle = "#1e293b"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"

      ctx.fillText("X₁", padding + plotSize / 2, padding + plotSize + 20)

      ctx.save()
      ctx.translate(padding - 20, padding + plotSize / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText("X₂", 0, 0)
      ctx.restore()

      // Draw training data points (XOR)
      const trainingData = [
        { x: 0, y: 0, label: 0 },
        { x: 0, y: 1, label: 1 },
        { x: 1, y: 0, label: 1 },
        { x: 1, y: 1, label: 0 },
      ]

      for (const point of trainingData) {
        const x = padding + point.x * plotSize
        const y = padding + (1 - point.y) * plotSize // Invert y for canvas coordinates

        ctx.beginPath()
        ctx.arc(x, y, 8, 0, Math.PI * 2)
        ctx.fillStyle = point.label === 0 ? "#10b981" : "#ef4444"
        ctx.fill()
        ctx.strokeStyle = "#1e293b"
        ctx.lineWidth = 1
        ctx.stroke()

        ctx.fillStyle = "#ffffff"
        ctx.font = "10px Arial"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(point.label.toString(), x, y)
      }

      // Draw test point
      const testX = padding + testPoint[0] * plotSize
      const testY = padding + (1 - testPoint[1]) * plotSize // Invert y for canvas coordinates

      ctx.beginPath()
      ctx.arc(testX, testY, 10, 0, Math.PI * 2)

      const prediction = predict(testPoint[0], testPoint[1])
      const predictedClass = prediction < 0.5 ? 0 : 1

      ctx.fillStyle = predictedClass === 0 ? "#10b981" : "#ef4444"
      ctx.fill()
      ctx.strokeStyle = "#1e293b"
      ctx.lineWidth = 2
      ctx.stroke()

      // Draw prediction info
      ctx.fillStyle = "#1e293b"
      ctx.font = "16px Arial"
      ctx.textAlign = "left"
      ctx.textBaseline = "top"

      ctx.fillText(`Test Point: (${testPoint[0].toFixed(2)}, ${testPoint[1].toFixed(2)})`, padding, 20)
      ctx.fillText(`Prediction: ${prediction.toFixed(4)}`, padding + 250, 20)
      ctx.fillText(`Class: ${predictedClass}`, padding + 450, 20)

      // Draw network visualization
      drawNetworkVisualization(padding + plotSize + 50, padding + plotSize / 2, 200, 150)

      // Continue animation
      const frame = requestAnimationFrame(draw)
      setAnimationFrame(frame)
    }

    // Draw network visualization
    const drawNetworkVisualization = (x: number, y: number, width: number, height: number) => {
      const layers = [2, 4, 1] // Input, hidden, output
      const layerSpacing = width / (layers.length - 1)

      // Draw connections
      for (let l = 0; l < layers.length - 1; l++) {
        const startX = x + l * layerSpacing
        const endX = x + (l + 1) * layerSpacing

        for (let i = 0; i < layers[l]; i++) {
          const startY = y - height / 2 + (i + 0.5) * (height / layers[l])

          for (let j = 0; j < layers[l + 1]; j++) {
            const endY = y - height / 2 + (j + 0.5) * (height / layers[l + 1])

            // Get weight
            const weight = weights[l][j][i]

            // Color based on weight
            const weightColor =
              weight >= 0
                ? `rgba(16, 185, 129, ${Math.min(1, Math.abs(weight) * 0.3 + 0.2)})` // Green for positive
                : `rgba(239, 68, 68, ${Math.min(1, Math.abs(weight) * 0.3 + 0.2)})` // Red for negative

            ctx.beginPath()
            ctx.moveTo(startX, startY)
            ctx.lineTo(endX, endY)
            ctx.strokeStyle = weightColor
            ctx.lineWidth = Math.max(1, Math.abs(weight) * 0.5)
            ctx.stroke()
          }
        }
      }

      // Draw neurons
      for (let l = 0; l < layers.length; l++) {
        const layerX = x + l * layerSpacing

        for (let i = 0; i < layers[l]; i++) {
          const neuronY = y - height / 2 + (i + 0.5) * (height / layers[l])

          ctx.beginPath()
          ctx.arc(layerX, neuronY, 10, 0, Math.PI * 2)

          if (l === 0) {
            // Input layer - use test point values
            ctx.fillStyle = `rgba(16, 185, 129, ${l === 0 && i === 0 ? testPoint[0] * 0.8 + 0.2 : testPoint[1] * 0.8 + 0.2})`
          } else if (l === layers.length - 1) {
            // Output layer - use prediction
            const prediction = predict(testPoint[0], testPoint[1])
            ctx.fillStyle = `rgba(239, 68, 68, ${prediction * 0.8 + 0.2})`
          } else {
            // Hidden layer
            ctx.fillStyle = "#3b82f6"
          }

          ctx.fill()
          ctx.strokeStyle = "#1e293b"
          ctx.lineWidth = 1
          ctx.stroke()
        }
      }

      // Draw layer labels
      ctx.fillStyle = "#1e293b"
      ctx.font = "12px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "bottom"

      ctx.fillText("Input", x, y - height / 2 - 5)
      ctx.fillText("Hidden", x + layerSpacing, y - height / 2 - 5)
      ctx.fillText("Output", x + 2 * layerSpacing, y - height / 2 - 5)
    }

    // Handle mouse move for test point
    const handleMouseMove = (e: MouseEvent) => {
      if (e.buttons !== 1) return // Only update when mouse button is pressed

      const rect = canvas.getBoundingClientRect()
      const mouseX = e.clientX - rect.left
      const mouseY = e.clientY - rect.top

      const padding = 50
      const plotSize = Math.min(canvas.width, canvas.height) - padding * 2

      // Check if mouse is within plot area
      if (mouseX >= padding && mouseX <= padding + plotSize && mouseY >= padding && mouseY <= padding + plotSize) {
        const x = (mouseX - padding) / plotSize
        const y = 1 - (mouseY - padding) / plotSize // Invert y for logical coordinates

        setTestPoint([x, y])
      }
    }

    // Add event listeners
    canvas.addEventListener("mousemove", handleMouseMove)
    canvas.addEventListener("mousedown", handleMouseMove)

    // Start animation
    const frame = requestAnimationFrame(draw)
    setAnimationFrame(frame)

    // Cleanup
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
      canvas.removeEventListener("mousemove", handleMouseMove)
      canvas.removeEventListener("mousedown", handleMouseMove)
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [testPoint, showDecisionBoundary])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center space-x-2">
          <label>Test Point:</label>
          <span>
            ({testPoint[0].toFixed(2)}, {testPoint[1].toFixed(2)})
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={showDecisionBoundary}
            onChange={(e) => setShowDecisionBoundary(e.target.checked)}
            id="show-boundary"
          />
          <label htmlFor="show-boundary">Show Decision Boundary</label>
        </div>

        <button
          onClick={() => setTestPoint([Math.random(), Math.random()])}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Random Test Point
        </button>
      </div>

      <canvas ref={canvasRef} className="flex-1 w-full border rounded cursor-crosshair" />

      <div className="mt-2 text-sm text-gray-600">
        Click and drag on the plot to set a test point and see the network's prediction
      </div>
    </div>
  )
}

