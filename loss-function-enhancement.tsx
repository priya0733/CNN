"use client"

import { useEffect, useRef } from "react"

export default function LossFunctionEnhancement() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    // This function would be added to loss_function.js
    const enhanceLossFunction = () => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      // Loss function parameters
      let lossType = "mse"
      let dataPoints = 10
      let noiseLevel = 0.2

      // Generate random data
      const generateData = (count, noise) => {
        const data = []
        // Generate a simple linear relationship with noise
        for (let i = 0; i < count; i++) {
          const x = (i / (count - 1)) * 2 - 1 // Range from -1 to 1
          const trueY = 0.5 * x + 0.2 // True relationship
          const noisyY = trueY + (Math.random() * 2 - 1) * noise // Add noise
          data.push({ x, y: noisyY })
        }
        return data
      }

      // Calculate loss for a given prediction
      const calculateLoss = (actual, predicted, type) => {
        switch (type) {
          case "mse":
            return Math.pow(actual - predicted, 2)
          case "mae":
            return Math.abs(actual - predicted)
          case "cross_entropy":
            // Simplified cross-entropy for demo
            const p = Math.max(0.001, Math.min(0.999, predicted)) // Clamp to avoid log(0)
            return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p))
          default:
            return Math.pow(actual - predicted, 2)
        }
      }

      // Draw the loss function visualization
      const drawLossVisualization = () => {
        const width = canvas.width
        const height = canvas.height

        // Clear canvas
        ctx.clearRect(0, 0, width, height)

        // Generate data
        const data = generateData(dataPoints, noiseLevel)

        // Draw axes
        ctx.beginPath()
        ctx.strokeStyle = "#ccc"
        ctx.lineWidth = 1
        ctx.moveTo(50, height - 50)
        ctx.lineTo(width - 50, height - 50) // x-axis
        ctx.moveTo(50, height - 50)
        ctx.lineTo(50, 50) // y-axis
        ctx.stroke()

        // Draw axis labels
        ctx.fillStyle = "black"
        ctx.font = "12px Arial"
        ctx.fillText("Input", width / 2, height - 20)
        ctx.save()
        ctx.translate(20, height / 2)
        ctx.rotate(-Math.PI / 2)
        ctx.fillText("Output", 0, 0)
        ctx.restore()

        // Calculate model parameters (simple linear regression)
        let sumX = 0,
          sumY = 0,
          sumXY = 0,
          sumX2 = 0
        for (const point of data) {
          sumX += point.x
          sumY += point.y
          sumXY += point.x * point.y
          sumX2 += point.x * point.x
        }

        const n = data.length
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        const intercept = (sumY - slope * sumX) / n

        // Draw the regression line
        ctx.beginPath()
        ctx.strokeStyle = "#3b82f6"
        ctx.lineWidth = 2

        const x1 = -1
        const y1 = slope * x1 + intercept
        const x2 = 1
        const y2 = slope * x2 + intercept

        // Convert to canvas coordinates
        const canvasX1 = ((x1 + 1) / 2) * (width - 100) + 50
        const canvasY1 = height - 50 - ((y1 + 1) / 2) * (height - 100)
        const canvasX2 = ((x2 + 1) / 2) * (width - 100) + 50
        const canvasY2 = height - 50 - ((y2 + 1) / 2) * (height - 100)

        ctx.moveTo(canvasX1, canvasY1)
        ctx.lineTo(canvasX2, canvasY2)
        ctx.stroke()

        // Draw data points
        for (const point of data) {
          const canvasX = ((point.x + 1) / 2) * (width - 100) + 50
          const canvasY = height - 50 - ((point.y + 1) / 2) * (height - 100)

          ctx.beginPath()
          ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI)
          ctx.fillStyle = "#f87171"
          ctx.fill()

          // Draw line to regression line (error)
          const predictedY = slope * point.x + intercept
          const canvasPredictedY = height - 50 - ((predictedY + 1) / 2) * (height - 100)

          ctx.beginPath()
          ctx.moveTo(canvasX, canvasY)
          ctx.lineTo(canvasX, canvasPredictedY)
          ctx.strokeStyle = "#ef4444"
          ctx.lineWidth = 1
          ctx.stroke()

          // Visualize loss
          const loss = calculateLoss(point.y, predictedY, lossType)
          const lossSize = Math.min(20, Math.max(5, loss * 30))

          ctx.beginPath()
          ctx.arc(canvasX, canvasPredictedY, lossSize, 0, 2 * Math.PI)
          ctx.fillStyle = "rgba(239, 68, 68, 0.2)"
          ctx.fill()
        }

        // Calculate and display total loss
        let totalLoss = 0
        for (const point of data) {
          const predictedY = slope * point.x + intercept
          totalLoss += calculateLoss(point.y, predictedY, lossType)
        }

        ctx.fillStyle = "black"
        ctx.font = "14px Arial"
        ctx.fillText(`${lossType.toUpperCase()} Loss: ${totalLoss.toFixed(3)}`, width - 200, 30)
      }

      // Initial draw
      drawLossVisualization()

      // Add event listeners for controls
      document.getElementById("loss-type")?.addEventListener("change", (e) => {
        lossType = (e.target as HTMLSelectElement).value
        drawLossVisualization()
      })

      document.getElementById("data-points-slider")?.addEventListener("input", (e) => {
        dataPoints = Number.parseInt((e.target as HTMLInputElement).value)
        document.getElementById("data-points-value").textContent = dataPoints.toString()
        drawLossVisualization()
      })

      document.getElementById("noise-slider")?.addEventListener("input", (e) => {
        noiseLevel = Number.parseFloat((e.target as HTMLInputElement).value)
        document.getElementById("noise-value").textContent = noiseLevel.toFixed(2)
        drawLossVisualization()
      })

      document.getElementById("regenerate-data")?.addEventListener("click", () => {
        drawLossVisualization()
      })
    }

    enhanceLossFunction()

    return () => {
      // Cleanup
    }
  }, [])

  return (
    <div className="p-4 border rounded-lg bg-white">
      <h2 className="text-xl font-semibold mb-4">Enhanced Loss Function Visualization</h2>

      <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label htmlFor="loss-type" className="block text-sm font-medium mb-1">
            Loss Function
          </label>
          <select id="loss-type" className="w-full p-2 border rounded">
            <option value="mse">Mean Squared Error</option>
            <option value="mae">Mean Absolute Error</option>
            <option value="cross_entropy">Cross Entropy</option>
          </select>
        </div>

        <div>
          <label htmlFor="data-points-slider" className="block text-sm font-medium mb-1">
            Data Points
          </label>
          <input id="data-points-slider" type="range" min="5" max="30" defaultValue="10" className="w-full" />
          <div className="flex justify-between text-sm text-gray-600 mt-1">
            <span>5</span>
            <span id="data-points-value">10</span>
            <span>30</span>
          </div>
        </div>

        <div>
          <label htmlFor="noise-slider" className="block text-sm font-medium mb-1">
            Noise Level
          </label>
          <input id="noise-slider" type="range" min="0" max="0.5" step="0.05" defaultValue="0.2" className="w-full" />
          <div className="flex justify-between text-sm text-gray-600 mt-1">
            <span>0.00</span>
            <span id="noise-value">0.20</span>
            <span>0.50</span>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <button id="regenerate-data" className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
          Regenerate Data
        </button>
      </div>

      <div className="border p-2 bg-gray-50 rounded">
        <canvas ref={canvasRef} width="600" height="300" className="w-full h-auto"></canvas>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          This visualization shows how different loss functions measure the error between predicted values (blue line)
          and actual data points (red dots). The size of the red circles represents the magnitude of the loss at each
          point.
        </p>
      </div>
    </div>
  )
}

