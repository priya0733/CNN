"use client"

import { useEffect, useRef } from "react"

export default function ActivationFunctionEnhancement() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    // This function would be added to activation_function.js
    const enhanceActivationFunction = () => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Add interactive elements to the existing activation function visualization
      // 1. Add sliders for parameters (alpha for leaky ReLU, etc.)
      // 2. Add real-time updating of the visualization
      // 3. Add tooltips showing values at different points

      // Sample code to draw an interactive activation function
      const drawActivationFunction = (type, params) => {
        const width = canvas.width
        const height = canvas.height
        const centerY = height / 2

        ctx.beginPath()
        ctx.moveTo(0, centerY)
        ctx.lineTo(width, centerY)
        ctx.strokeStyle = "#ccc"
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(width / 2, 0)
        ctx.lineTo(width / 2, height)
        ctx.strokeStyle = "#ccc"
        ctx.stroke()

        // Draw the activation function
        ctx.beginPath()
        ctx.strokeStyle = "#3b82f6"
        ctx.lineWidth = 2

        for (let x = 0; x < width; x++) {
          const normalizedX = (x - width / 2) / 50
          let y

          switch (type) {
            case "relu":
              y = Math.max(0, normalizedX)
              break
            case "sigmoid":
              y = 1 / (1 + Math.exp(-normalizedX))
              break
            case "tanh":
              y = Math.tanh(normalizedX)
              break
            case "leaky_relu":
              y = normalizedX > 0 ? normalizedX : params.alpha * normalizedX
              break
            default:
              y = normalizedX
          }

          // Scale and invert y for canvas coordinates
          const canvasY = centerY - y * 50

          if (x === 0) {
            ctx.moveTo(x, canvasY)
          } else {
            ctx.lineTo(x, canvasY)
          }
        }

        ctx.stroke()

        // Add interactive point that follows mouse
        canvas.onmousemove = (e) => {
          const rect = canvas.getBoundingClientRect()
          const mouseX = e.clientX - rect.left
          const normalizedX = (mouseX - width / 2) / 50

          let y
          switch (type) {
            case "relu":
              y = Math.max(0, normalizedX)
              break
            case "sigmoid":
              y = 1 / (1 + Math.exp(-normalizedX))
              break
            case "tanh":
              y = Math.tanh(normalizedX)
              break
            case "leaky_relu":
              y = normalizedX > 0 ? normalizedX : params.alpha * normalizedX
              break
            default:
              y = normalizedX
          }

          // Redraw to clear previous tooltip
          drawActivationFunction(type, params)

          // Draw point at mouse position
          const canvasY = centerY - y * 50
          ctx.beginPath()
          ctx.arc(mouseX, canvasY, 5, 0, 2 * Math.PI)
          ctx.fillStyle = "red"
          ctx.fill()

          // Draw tooltip
          ctx.fillStyle = "black"
          ctx.font = "12px Arial"
          ctx.fillText(`x: ${normalizedX.toFixed(2)}, y: ${y.toFixed(2)}`, mouseX + 10, canvasY - 10)
        }
      }

      // Initial draw with default parameters
      drawActivationFunction("relu", { alpha: 0.1 })

      // Add event listeners for parameter changes
      document.getElementById("function-type")?.addEventListener("change", (e) => {
        const type = (e.target as HTMLSelectElement).value
        const alpha = Number.parseFloat((document.getElementById("alpha-slider") as HTMLInputElement).value)
        drawActivationFunction(type, { alpha })
      })

      document.getElementById("alpha-slider")?.addEventListener("input", (e) => {
        const alpha = Number.parseFloat((e.target as HTMLInputElement).value)
        const type = (document.getElementById("function-type") as HTMLSelectElement).value
        drawActivationFunction(type, { alpha })
      })
    }

    enhanceActivationFunction()

    // Cleanup
    return () => {
      if (canvasRef.current) {
        canvasRef.current.onmousemove = null
      }
    }
  }, [])

  return (
    <div className="p-4 border rounded-lg bg-white">
      <h2 className="text-xl font-semibold mb-4">Enhanced Activation Function Visualization</h2>

      <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label htmlFor="function-type" className="block text-sm font-medium mb-1">
            Function Type
          </label>
          <select id="function-type" className="w-full p-2 border rounded">
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="leaky_relu">Leaky ReLU</option>
          </select>
        </div>

        <div>
          <label htmlFor="alpha-slider" className="block text-sm font-medium mb-1">
            Alpha (for Leaky ReLU)
          </label>
          <input id="alpha-slider" type="range" min="0" max="0.5" step="0.01" defaultValue="0.1" className="w-full" />
          <div id="alpha-value" className="text-sm text-gray-600 mt-1">
            0.1
          </div>
        </div>
      </div>

      <div className="border p-2 bg-gray-50 rounded">
        <canvas ref={canvasRef} width="600" height="300" className="w-full h-auto"></canvas>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          Move your mouse over the graph to see exact values. Adjust parameters to see how they affect the activation
          function.
        </p>
      </div>
    </div>
  )
}

