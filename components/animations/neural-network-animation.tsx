"use client"

import { useEffect, useRef, useState } from "react"

export default function NeuralNetworkAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
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
    const layers = [4, 6, 5, 3]
    const neurons: { x: number; y: number; layer: number; index: number }[] = []
    const connections: { from: number; to: number }[] = []
    const signals: { x: number; y: number; targetX: number; targetY: number; progress: number; color: string }[] = []

    // Initialize neurons
    const initializeNetwork = () => {
      neurons.length = 0
      connections.length = 0

      const padding = 50
      const layerSpacing = (canvas.width - padding * 2) / (layers.length - 1)

      // Create neurons
      let neuronIndex = 0
      for (let l = 0; l < layers.length; l++) {
        const neuronsInLayer = layers[l]
        const layerHeight = neuronsInLayer * 40
        const startY = (canvas.height - layerHeight) / 2

        for (let n = 0; n < neuronsInLayer; n++) {
          neurons.push({
            x: padding + l * layerSpacing,
            y: startY + n * 40 + 20,
            layer: l,
            index: neuronIndex++,
          })
        }
      }

      // Create connections
      for (let l = 0; l < layers.length - 1; l++) {
        const startNeurons = neurons.filter((n) => n.layer === l)
        const endNeurons = neurons.filter((n) => n.layer === l + 1)

        for (const start of startNeurons) {
          for (const end of endNeurons) {
            connections.push({ from: start.index, to: end.index })
          }
        }
      }
    }

    // Create a new signal
    const createSignal = () => {
      // Randomly select an input neuron
      const inputNeurons = neurons.filter((n) => n.layer === 0)
      const randomInputIndex = Math.floor(Math.random() * inputNeurons.length)
      const startNeuron = inputNeurons[randomInputIndex]

      // Find all connections from this neuron
      const outgoingConnections = connections.filter((c) => c.from === startNeuron.index)

      // Create signals for each connection
      for (const conn of outgoingConnections) {
        const endNeuron = neurons.find((n) => n.index === conn.to)
        if (endNeuron) {
          const colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
          signals.push({
            x: startNeuron.x,
            y: startNeuron.y,
            targetX: endNeuron.x,
            targetY: endNeuron.y,
            progress: 0,
            color: colors[Math.floor(Math.random() * colors.length)],
          })
        }
      }
    }

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw connections
      ctx.strokeStyle = "#e2e8f0"
      ctx.lineWidth = 1

      for (const conn of connections) {
        const start = neurons.find((n) => n.index === conn.from)
        const end = neurons.find((n) => n.index === conn.to)

        if (start && end) {
          ctx.beginPath()
          ctx.moveTo(start.x, start.y)
          ctx.lineTo(end.x, end.y)
          ctx.stroke()
        }
      }

      // Draw neurons
      for (const neuron of neurons) {
        ctx.beginPath()
        ctx.arc(neuron.x, neuron.y, 10, 0, Math.PI * 2)

        // Different colors for different layers
        if (neuron.layer === 0) {
          ctx.fillStyle = "#10b981" // Input layer - green
        } else if (neuron.layer === layers.length - 1) {
          ctx.fillStyle = "#ef4444" // Output layer - red
        } else {
          ctx.fillStyle = "#3b82f6" // Hidden layers - blue
        }

        ctx.fill()
        ctx.strokeStyle = "#1e293b"
        ctx.lineWidth = 1
        ctx.stroke()
      }

      // Update and draw signals
      for (let i = 0; i < signals.length; i++) {
        const signal = signals[i]

        // Update position
        signal.progress += 0.01
        const x = signal.x + (signal.targetX - signal.x) * signal.progress
        const y = signal.y + (signal.targetY - signal.y) * signal.progress

        // Draw signal
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, Math.PI * 2)
        ctx.fillStyle = signal.color
        ctx.fill()

        // Remove completed signals
        if (signal.progress >= 1) {
          // Find the target neuron
          const targetNeuron = neurons.find((n) => n.x === signal.targetX && n.y === signal.targetY)

          // If this is not the output layer, create new signals
          if (targetNeuron && targetNeuron.layer < layers.length - 1) {
            // Find all connections from this neuron
            const outgoingConnections = connections.filter((c) => c.from === targetNeuron.index)

            // Create signals for each connection
            for (const conn of outgoingConnections) {
              const endNeuron = neurons.find((n) => n.index === conn.to)
              if (endNeuron) {
                signals.push({
                  x: targetNeuron.x,
                  y: targetNeuron.y,
                  targetX: endNeuron.x,
                  targetY: endNeuron.y,
                  progress: 0,
                  color: signal.color,
                })
              }
            }
          }

          signals.splice(i, 1)
          i--
        }
      }

      // Occasionally create new signals
      if (Math.random() < 0.02 && signals.length < 50) {
        createSignal()
      }

      // Continue animation
      const frame = requestAnimationFrame(animate)
      setAnimationFrame(frame)
    }

    // Initialize and start animation
    initializeNetwork()
    const frame = requestAnimationFrame(animate)
    setAnimationFrame(frame)

    // Cleanup
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [])

  return <canvas ref={canvasRef} className="w-full h-full" />
}

