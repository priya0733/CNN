"use client"

import { useEffect, useRef } from "react"

export default function NetworkAnimationEnhancement() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    // This function would be added to network_animation.js
    const enhanceNetworkAnimation = () => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      // Network parameters
      let layers = 3
      let neuronsPerLayer = 4
      let animationSpeed = 1
      let animationRunning = true

      // Animation variables
      let signals = []
      let animationFrame

      // Draw the neural network
      const drawNetwork = () => {
        const width = canvas.width
        const height = canvas.height

        // Clear canvas
        ctx.clearRect(0, 0, width, height)

        // Calculate positions
        const layerGap = width / (layers + 1)
        const neuronGap = height / (neuronsPerLayer + 1)

        // Draw connections first (so they appear behind neurons)
        ctx.strokeStyle = "#ccc"
        ctx.lineWidth = 1

        for (let l = 0; l < layers - 1; l++) {
          const x1 = (l + 1) * layerGap
          const x2 = (l + 2) * layerGap

          for (let n1 = 0; n1 < neuronsPerLayer; n1++) {
            const y1 = (n1 + 1) * neuronGap

            for (let n2 = 0; n2 < neuronsPerLayer; n2++) {
              const y2 = (n2 + 1) * neuronGap

              ctx.beginPath()
              ctx.moveTo(x1, y1)
              ctx.lineTo(x2, y2)
              ctx.stroke()
            }
          }
        }

        // Draw neurons
        for (let l = 0; l < layers; l++) {
          const x = (l + 1) * layerGap

          for (let n = 0; n < neuronsPerLayer; n++) {
            const y = (n + 1) * neuronGap

            ctx.beginPath()
            ctx.arc(x, y, 15, 0, 2 * Math.PI)
            ctx.fillStyle = l === 0 ? "#4ade80" : l === layers - 1 ? "#f87171" : "#60a5fa"
            ctx.fill()
            ctx.strokeStyle = "#333"
            ctx.lineWidth = 1
            ctx.stroke()
          }
        }

        // Draw signals
        ctx.lineWidth = 3
        for (let i = 0; i < signals.length; i++) {
          const signal = signals[i]

          // Calculate position along the connection
          const startX = (signal.fromLayer + 1) * layerGap
          const startY = (signal.fromNeuron + 1) * neuronGap
          const endX = (signal.toLayer + 1) * layerGap
          const endY = (signal.toNeuron + 1) * neuronGap

          const progress = signal.progress
          const x = startX + (endX - startX) * progress
          const y = startY + (endY - startY) * progress

          // Draw signal
          ctx.beginPath()
          ctx.arc(x, y, 5, 0, 2 * Math.PI)
          ctx.fillStyle = signal.color
          ctx.fill()

          // Update progress
          signal.progress += 0.02 * animationSpeed

          // Remove signals that have reached their destination
          if (signal.progress >= 1) {
            signals.splice(i, 1)
            i--

            // Create new signals if this signal reached a hidden layer
            if (signal.toLayer < layers - 2) {
              for (let n = 0; n < neuronsPerLayer; n++) {
                createSignal(signal.toLayer, signal.toNeuron, signal.toLayer + 1, n)
              }
            }
          }
        }

        // Add new signals from input layer if needed
        if (signals.length === 0 && animationRunning) {
          const inputNeuron = Math.floor(Math.random() * neuronsPerLayer)
          for (let n = 0; n < neuronsPerLayer; n++) {
            createSignal(0, inputNeuron, 1, n)
          }
        }

        // Continue animation
        animationFrame = requestAnimationFrame(drawNetwork)
      }

      // Create a new signal
      const createSignal = (fromLayer, fromNeuron, toLayer, toNeuron) => {
        const colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
        signals.push({
          fromLayer,
          fromNeuron,
          toLayer,
          toNeuron,
          progress: 0,
          color: colors[Math.floor(Math.random() * colors.length)],
        })
      }

      // Start animation
      drawNetwork()

      // Add event listeners for controls
      document.getElementById("layers-slider")?.addEventListener("input", (e) => {
        layers = Number.parseInt((e.target as HTMLInputElement).value)
        document.getElementById("layers-value").textContent = layers.toString()
        signals = [] // Reset signals when network changes
      })

      document.getElementById("neurons-slider")?.addEventListener("input", (e) => {
        neuronsPerLayer = Number.parseInt((e.target as HTMLInputElement).value)
        document.getElementById("neurons-value").textContent = neuronsPerLayer.toString()
        signals = [] // Reset signals when network changes
      })

      document.getElementById("speed-slider")?.addEventListener("input", (e) => {
        animationSpeed = Number.parseFloat((e.target as HTMLInputElement).value)
        document.getElementById("speed-value").textContent = animationSpeed.toFixed(1) + "x"
      })

      document.getElementById("toggle-animation")?.addEventListener("click", () => {
        animationRunning = !animationRunning
        ;(document.getElementById("toggle-animation") as HTMLButtonElement).textContent = animationRunning
          ? "Pause"
          : "Resume"
      })

      // Cleanup
      return () => {
        cancelAnimationFrame(animationFrame)
      }
    }

    enhanceNetworkAnimation()

    return () => {
      if (canvasRef.current) {
        // Cleanup event listeners
      }
    }
  }, [])

  return (
    <div className="p-4 border rounded-lg bg-white">
      <h2 className="text-xl font-semibold mb-4">Enhanced Network Animation</h2>

      <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label htmlFor="layers-slider" className="block text-sm font-medium mb-1">
            Layers
          </label>
          <input id="layers-slider" type="range" min="2" max="5" defaultValue="3" className="w-full" />
          <div className="flex justify-between text-sm text-gray-600 mt-1">
            <span>2</span>
            <span id="layers-value">3</span>
            <span>5</span>
          </div>
        </div>

        <div>
          <label htmlFor="neurons-slider" className="block text-sm font-medium mb-1">
            Neurons per Layer
          </label>
          <input id="neurons-slider" type="range" min="2" max="8" defaultValue="4" className="w-full" />
          <div className="flex justify-between text-sm text-gray-600 mt-1">
            <span>2</span>
            <span id="neurons-value">4</span>
            <span>8</span>
          </div>
        </div>

        <div>
          <label htmlFor="speed-slider" className="block text-sm font-medium mb-1">
            Animation Speed
          </label>
          <input id="speed-slider" type="range" min="0.5" max="3" step="0.5" defaultValue="1" className="w-full" />
          <div className="flex justify-between text-sm text-gray-600 mt-1">
            <span>0.5x</span>
            <span id="speed-value">1.0x</span>
            <span>3.0x</span>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <button id="toggle-animation" className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
          Pause
        </button>
      </div>

      <div className="border p-2 bg-gray-50 rounded">
        <canvas ref={canvasRef} width="600" height="300" className="w-full h-auto"></canvas>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          Adjust the sliders to change the network architecture and animation speed. The animation shows how data flows
          through the neural network.
        </p>
      </div>
    </div>
  )
}

