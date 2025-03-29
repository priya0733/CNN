"use client"

import { useEffect, useRef, useState } from "react"

export default function NetworkBuildAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [layers, setLayers] = useState(3)
  const [neuronsPerLayer, setNeuronsPerLayer] = useState(4)
  const [animationFrame, setAnimationFrame] = useState<number | null>(null)
  const [buildProgress, setBuildProgress] = useState(0)

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

    // Animation variables
    let progress = buildProgress
    let animating = true

    // Draw the network
    const drawNetwork = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Calculate positions
      const padding = 50
      const availableWidth = canvas.width - padding * 2
      const availableHeight = canvas.height - padding * 2

      const layerSpacing = availableWidth / (layers - 1)
      const neuronSpacing = Math.min(30, availableHeight / (neuronsPerLayer + 1))
      const neuronRadius = Math.min(10, neuronSpacing / 3)

      // Draw connections and neurons up to the current progress
      const currentLayer = Math.floor(progress * layers)
      const layerProgress = (progress * layers) % 1

      // Draw connections between completed layers
      for (let l = 0; l < currentLayer; l++) {
        if (l < currentLayer - 1) {
          const x1 = padding + l * layerSpacing
          const x2 = padding + (l + 1) * layerSpacing

          for (let n1 = 0; n1 < neuronsPerLayer; n1++) {
            const y1 = padding + (n1 + 1) * neuronSpacing

            for (let n2 = 0; n2 < neuronsPerLayer; n2++) {
              const y2 = padding + (n2 + 1) * neuronSpacing

              ctx.beginPath()
              ctx.moveTo(x1, y1)
              ctx.lineTo(x2, y2)
              ctx.strokeStyle = "#cbd5e1"
              ctx.lineWidth = 1
              ctx.stroke()
            }
          }
        }
      }

      // Draw connections for the current layer in progress
      if (currentLayer < layers - 1) {
        const x1 = padding + currentLayer * layerSpacing
        const x2 = padding + (currentLayer + 1) * layerSpacing

        for (let n1 = 0; n1 < neuronsPerLayer; n1++) {
          const y1 = padding + (n1 + 1) * neuronSpacing

          for (let n2 = 0; n2 < neuronsPerLayer; n2++) {
            const y2 = padding + (n2 + 1) * neuronSpacing

            // Only draw connections up to the current progress
            if (n1 / neuronsPerLayer <= layerProgress) {
              ctx.beginPath()
              ctx.moveTo(x1, y1)

              // Animate the connection drawing
              const connectionProgress = Math.max(
                0,
                Math.min(1, (layerProgress - n1 / neuronsPerLayer) * neuronsPerLayer),
              )
              const endX = x1 + (x2 - x1) * connectionProgress
              const endY = y1 + (y2 - y1) * connectionProgress

              ctx.lineTo(endX, endY)
              ctx.strokeStyle = "#3b82f6"
              ctx.lineWidth = 1
              ctx.stroke()
            }
          }
        }
      }

      // Draw neurons
      for (let l = 0; l < layers; l++) {
        const x = padding + l * layerSpacing

        for (let n = 0; n < neuronsPerLayer; n++) {
          const y = padding + (n + 1) * neuronSpacing

          // Only draw neurons up to the current progress
          if (l < currentLayer || (l === currentLayer && n / neuronsPerLayer <= layerProgress)) {
            ctx.beginPath()
            ctx.arc(x, y, neuronRadius, 0, Math.PI * 2)

            // Different colors for different layers
            if (l === 0) {
              ctx.fillStyle = "#10b981" // Input layer - green
            } else if (l === layers - 1) {
              ctx.fillStyle = "#ef4444" // Output layer - red
            } else {
              ctx.fillStyle = "#3b82f6" // Hidden layers - blue
            }

            ctx.fill()
            ctx.strokeStyle = "#1e293b"
            ctx.lineWidth = 1
            ctx.stroke()
          }
        }
      }

      // Draw layer labels
      ctx.fillStyle = "#1e293b"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"

      if (currentLayer >= 0) {
        ctx.fillText("Input Layer", padding, padding / 2)
      }

      for (let l = 1; l < layers - 1; l++) {
        if (l <= currentLayer) {
          ctx.fillText(`Hidden Layer ${l}`, padding + l * layerSpacing, padding / 2)
        }
      }

      if (currentLayer >= layers - 1) {
        ctx.fillText("Output Layer", padding + (layers - 1) * layerSpacing, padding / 2)
      }

      // Update progress
      if (animating) {
        progress += 0.005
        if (progress >= 1) {
          progress = 1
          animating = false
        }
        setBuildProgress(progress)
      }

      // Continue animation
      const frame = requestAnimationFrame(drawNetwork)
      setAnimationFrame(frame)
    }

    // Start animation
    const frame = requestAnimationFrame(drawNetwork)
    setAnimationFrame(frame)

    // Cleanup
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [layers, neuronsPerLayer, buildProgress])

  // Reset animation when parameters change
  useEffect(() => {
    setBuildProgress(0)
  }, [layers, neuronsPerLayer])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center space-x-2">
          <label>Layers:</label>
          <input
            type="range"
            min="2"
            max="5"
            value={layers}
            onChange={(e) => setLayers(Number.parseInt(e.target.value))}
            className="w-32"
          />
          <span>{layers}</span>
        </div>

        <div className="flex items-center space-x-2">
          <label>Neurons per layer:</label>
          <input
            type="range"
            min="2"
            max="8"
            value={neuronsPerLayer}
            onChange={(e) => setNeuronsPerLayer(Number.parseInt(e.target.value))}
            className="w-32"
          />
          <span>{neuronsPerLayer}</span>
        </div>

        <button
          onClick={() => setBuildProgress(0)}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Rebuild
        </button>
      </div>

      <canvas ref={canvasRef} className="flex-1 w-full border rounded" />
    </div>
  )
}

