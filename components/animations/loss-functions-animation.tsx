"use client"

import { useEffect, useRef, useState } from "react"

export default function LossFunctionsAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [lossType, setLossType] = useState("mse")
  const [dataPoints, setDataPoints] = useState(10)
  const [noiseLevel, setNoiseLevel] = useState(0.2)

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

    // Generate random data
    const generateData = () => {
      const data = []

      // Generate a simple linear relationship with noise
      for (let i = 0; i < dataPoints; i++) {
        const x = (i / (dataPoints - 1)) * 2 - 1 // Range from -1 to 1
        const trueY = 0.5 * x + 0.2 // True relationship
        const noisyY = trueY + (Math.random() * 2 - 1) * noiseLevel // Add noise
        data.push({ x, y: noisyY })
      }

      return data
    }

    // Loss functions
    const lossFunctions = {
      mse: (actual: number, predicted: number) => Math.pow(actual - predicted, 2),
      mae: (actual: number, predicted: number) => Math.abs(actual - predicted),
      huber: (actual: number, predicted: number, delta = 1) => {
        const error = Math.abs(actual - predicted)
        return error <= delta ? 0.5 * Math.pow(error, 2) : delta * (error - 0.5 * delta)
      },
      crossEntropy: (actual: number, predicted: number) => {
        // Clip predicted to avoid log(0)
        const p = Math.max(0.001, Math.min(0.999, predicted))
        return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p))
      },
    }

    // Generate data
    const data = generateData()

    // Calculate linear regression parameters
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

    // Draw function
    const drawLossVisualization = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw axes
      const padding = 40
      const plotWidth = canvas.width - padding * 2
      const plotHeight = canvas.height - padding * 2

      ctx.strokeStyle = "#cbd5e1"
      ctx.lineWidth = 1

      // X-axis
      ctx.beginPath()
      ctx.moveTo(padding, canvas.height - padding)
      ctx.lineTo(canvas.width - padding, canvas.height - padding)
      ctx.stroke()

      // Y-axis
      ctx.beginPath()
      ctx.moveTo(padding, padding)
      ctx.lineTo(padding, canvas.height - padding)
      ctx.stroke()

      // Draw axis labels
      ctx.fillStyle = "#1e293b"
      ctx.font = "12px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("Input", canvas.width / 2, canvas.height - padding / 2)

      ctx.save()
      ctx.translate(padding / 2, canvas.height / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText("Output", 0, 0)
      ctx.restore()

      // Draw regression line
      ctx.strokeStyle = "#3b82f6"
      ctx.lineWidth = 2
      ctx.beginPath()

      const x1 = -1
      const y1 = slope * x1 + intercept
      const x2 = 1
      const y2 = slope * x2 + intercept

      // Convert to canvas coordinates
      const canvasX1 = padding + ((x1 + 1) / 2) * plotWidth
      const canvasY1 = canvas.height - padding - ((y1 + 1) / 2) * plotHeight
      const canvasX2 = padding + ((x2 + 1) / 2) * plotWidth
      const canvasY2 = canvas.height - padding - ((y2 + 1) / 2) * plotHeight

      ctx.moveTo(canvasX1, canvasY1)
      ctx.lineTo(canvasX2, canvasY2)
      ctx.stroke()

      // Draw data points and loss
      let totalLoss = 0

      for (const point of data) {
        // Convert to canvas coordinates
        const canvasX = padding + ((point.x + 1) / 2) * plotWidth
        const canvasY = canvas.height - padding - ((point.y + 1) / 2) * plotHeight

        // Draw data point
        ctx.beginPath()
        ctx.arc(canvasX, canvasY, 5, 0, Math.PI * 2)
        ctx.fillStyle = "#ef4444"
        ctx.fill()

        // Calculate predicted value
        const predictedY = slope * point.x + intercept
        const canvasPredictedY = canvas.height - padding - ((predictedY + 1) / 2) * plotHeight

        // Draw line to regression line (error)
        ctx.beginPath()
        ctx.moveTo(canvasX, canvasY)
        ctx.lineTo(canvasX, canvasPredictedY)
        ctx.strokeStyle = "#ef4444"
        ctx.lineWidth = 1
        ctx.stroke()

        // Calculate loss
        let loss
        switch (lossType) {
          case "mse":
            loss = lossFunctions.mse(point.y, predictedY)
            break
          case "mae":
            loss = lossFunctions.mae(point.y, predictedY)
            break
          case "huber":
            loss = lossFunctions.huber(point.y, predictedY)
            break
          case "crossEntropy":
            // Normalize values to [0,1] for cross entropy
            const normalizedActual = (point.y + 1) / 2
            const normalizedPredicted = (predictedY + 1) / 2
            loss = lossFunctions.crossEntropy(normalizedActual, normalizedPredicted)
            break
          default:
            loss = lossFunctions.mse(point.y, predictedY)
        }

        totalLoss += loss

        // Visualize loss
        const lossSize = Math.min(20, Math.max(5, loss * 30))

        ctx.beginPath()
        ctx.arc(canvasX, canvasPredictedY, lossSize, 0, Math.PI * 2)
        ctx.fillStyle = "rgba(239, 68, 68, 0.2)"
        ctx.fill()
      }

      // Draw loss type and total loss
      ctx.fillStyle = "#1e293b"
      ctx.font = "bold 16px Arial"
      ctx.textAlign = "left"
      ctx.textBaseline = "top"

      let lossName
      switch (lossType) {
        case "mse":
          lossName = "Mean Squared Error"
          break
        case "mae":
          lossName = "Mean Absolute Error"
          break
        case "huber":
          lossName = "Huber Loss"
          break
        case "crossEntropy":
          lossName = "Cross Entropy Loss"
          break
        default:
          lossName = "Mean Squared Error"
      }

      ctx.fillText(lossName, padding, padding / 2)
      ctx.font = "14px Arial"
      ctx.fillText(`Total Loss: ${totalLoss.toFixed(3)}`, canvas.width - 150, padding / 2)

      // Draw loss function formula
      ctx.font = "12px Arial"
      let formula
      switch (lossType) {
        case "mse":
          formula = "MSE = (1/n) Σ(y - ŷ)²"
          break
        case "mae":
          formula = "MAE = (1/n) Σ|y - ŷ|"
          break
        case "huber":
          formula = "Huber = (1/n) Σ(0.5(y - ŷ)² if |y - ŷ| ≤ δ, else δ(|y - ŷ| - 0.5δ))"
          break
        case "crossEntropy":
          formula = "CE = -(y log(ŷ) + (1-y) log(1-ŷ))"
          break
        default:
          formula = "MSE = (1/n) Σ(y - ŷ)²"
      }

      ctx.fillText(formula, padding, padding / 2 + 20)
    }

    // Initial draw
    drawLossVisualization()

    // Cleanup
    return () => {
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [lossType, dataPoints, noiseLevel])

  // Regenerate data
  const regenerateData = () => {
    // This will trigger a re-render with new random data
    setDataPoints(dataPoints)
    setNoiseLevel(noiseLevel)
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center space-x-2">
          <label>Loss Function:</label>
          <select value={lossType} onChange={(e) => setLossType(e.target.value)} className="px-2 py-1 border rounded">
            <option value="mse">Mean Squared Error</option>
            <option value="mae">Mean Absolute Error</option>
            <option value="huber">Huber Loss</option>
            <option value="crossEntropy">Cross Entropy</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <label>Data Points:</label>
          <input
            type="range"
            min="5"
            max="30"
            value={dataPoints}
            onChange={(e) => setDataPoints(Number.parseInt(e.target.value))}
            className="w-32"
          />
          <span>{dataPoints}</span>
        </div>

        <div className="flex items-center space-x-2">
          <label>Noise:</label>
          <input
            type="range"
            min="0"
            max="0.5"
            step="0.05"
            value={noiseLevel}
            onChange={(e) => setNoiseLevel(Number.parseFloat(e.target.value))}
            className="w-32"
          />
          <span>{noiseLevel.toFixed(2)}</span>
        </div>

        <button onClick={regenerateData} className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600">
          Regenerate Data
        </button>
      </div>

      <canvas ref={canvasRef} className="flex-1 w-full border rounded" />
    </div>
  )
}

