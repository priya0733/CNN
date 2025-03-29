"use client"

import { useEffect, useRef, useState } from "react"

export default function ActivationFunctionsAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [functionType, setFunctionType] = useState("sigmoid")
  const [alpha, setAlpha] = useState(0.1)
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number } | null>(null)

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

    // Activation functions
    const activationFunctions = {
      sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
      tanh: (x: number) => Math.tanh(x),
      relu: (x: number) => Math.max(0, x),
      leakyRelu: (x: number, alpha: number) => (x > 0 ? x : alpha * x),
      elu: (x: number, alpha: number) => (x > 0 ? x : alpha * (Math.exp(x) - 1)),
    }

    // Draw function
    const drawFunction = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw axes
      const originX = canvas.width / 2
      const originY = canvas.height / 2

      ctx.strokeStyle = "#cbd5e1"
      ctx.lineWidth = 1

      // X-axis
      ctx.beginPath()
      ctx.moveTo(0, originY)
      ctx.lineTo(canvas.width, originY)
      ctx.stroke()

      // Y-axis
      ctx.beginPath()
      ctx.moveTo(originX, 0)
      ctx.lineTo(originX, canvas.height)
      ctx.stroke()

      // Draw grid
      const gridSize = 50
      ctx.strokeStyle = "#e2e8f0"
      ctx.lineWidth = 0.5

      // Vertical grid lines
      for (let x = originX; x < canvas.width; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height)
        ctx.stroke()
      }

      for (let x = originX; x > 0; x -= gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height)
        ctx.stroke()
      }

      // Horizontal grid lines
      for (let y = originY; y < canvas.height; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      for (let y = originY; y > 0; y -= gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // Draw function
      ctx.strokeStyle = "#3b82f6"
      ctx.lineWidth = 3
      ctx.beginPath()

      for (let px = 0; px < canvas.width; px++) {
        // Convert pixel x to function input x
        const x = (px - originX) / gridSize

        // Calculate function output
        let y
        switch (functionType) {
          case "sigmoid":
            y = activationFunctions.sigmoid(x)
            break
          case "tanh":
            y = activationFunctions.tanh(x)
            break
          case "relu":
            y = activationFunctions.relu(x)
            break
          case "leakyRelu":
            y = activationFunctions.leakyRelu(x, alpha)
            break
          case "elu":
            y = activationFunctions.elu(x, alpha)
            break
          default:
            y = activationFunctions.sigmoid(x)
        }

        // Convert function output to pixel y
        const py = originY - y * gridSize

        if (px === 0) {
          ctx.moveTo(px, py)
        } else {
          ctx.lineTo(px, py)
        }
      }

      ctx.stroke()

      // Draw function name and formula
      ctx.fillStyle = "#1e293b"
      ctx.font = "bold 16px Arial"
      ctx.textAlign = "left"
      ctx.textBaseline = "top"

      let formula = ""
      switch (functionType) {
        case "sigmoid":
          formula = "σ(x) = 1 / (1 + e^(-x))"
          break
        case "tanh":
          formula = "tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"
          break
        case "relu":
          formula = "ReLU(x) = max(0, x)"
          break
        case "leakyRelu":
          formula = `Leaky ReLU(x) = x if x > 0, ${alpha}x otherwise`
          break
        case "elu":
          formula = `ELU(x) = x if x > 0, ${alpha}(e^x - 1) otherwise`
          break
      }

      ctx.fillText(functionType.charAt(0).toUpperCase() + functionType.slice(1), 20, 20)
      ctx.font = "14px Arial"
      ctx.fillText(formula, 20, 45)

      // Draw mouse position value if available
      if (mousePosition) {
        const x = (mousePosition.x - originX) / gridSize

        // Calculate function output
        let y
        switch (functionType) {
          case "sigmoid":
            y = activationFunctions.sigmoid(x)
            break
          case "tanh":
            y = activationFunctions.tanh(x)
            break
          case "relu":
            y = activationFunctions.relu(x)
            break
          case "leakyRelu":
            y = activationFunctions.leakyRelu(x, alpha)
            break
          case "elu":
            y = activationFunctions.elu(x, alpha)
            break
          default:
            y = activationFunctions.sigmoid(x)
        }

        // Draw point
        ctx.fillStyle = "#ef4444"
        ctx.beginPath()
        ctx.arc(mousePosition.x, originY - y * gridSize, 5, 0, Math.PI * 2)
        ctx.fill()

        // Draw coordinates
        ctx.fillStyle = "#1e293b"
        ctx.font = "14px Arial"
        ctx.textAlign = "left"
        ctx.textBaseline = "bottom"
        ctx.fillText(`x: ${x.toFixed(2)}, y: ${y.toFixed(2)}`, mousePosition.x + 10, originY - y * gridSize - 5)
      }
    }

    // Handle mouse move
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      setMousePosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      })
    }

    // Handle mouse leave
    const handleMouseLeave = () => {
      setMousePosition(null)
    }

    // Add event listeners
    canvas.addEventListener("mousemove", handleMouseMove)
    canvas.addEventListener("mouseleave", handleMouseLeave)

    // Initial draw
    drawFunction()

    // Redraw on changes
    const interval = setInterval(drawFunction, 30)

    // Cleanup
    return () => {
      clearInterval(interval)
      canvas.removeEventListener("mousemove", handleMouseMove)
      canvas.removeEventListener("mouseleave", handleMouseLeave)
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [functionType, alpha, mousePosition])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex justify-center space-x-4 mb-2">
        <select
          value={functionType}
          onChange={(e) => setFunctionType(e.target.value)}
          className="px-2 py-1 border rounded"
        >
          <option value="sigmoid">Sigmoid</option>
          <option value="tanh">Tanh</option>
          <option value="relu">ReLU</option>
          <option value="leakyRelu">Leaky ReLU</option>
          <option value="elu">ELU</option>
        </select>

        {(functionType === "leakyRelu" || functionType === "elu") && (
          <div className="flex items-center space-x-2">
            <label>Alpha:</label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={alpha}
              onChange={(e) => setAlpha(Number.parseFloat(e.target.value))}
              className="w-32"
            />
            <span>{alpha.toFixed(2)}</span>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="flex-1 w-full" />
    </div>
  )
}

