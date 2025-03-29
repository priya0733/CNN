"use client"

import { useEffect, useRef, useState } from "react"

export default function NetworkLayersAnimation() {
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
    const neurons: { x: number; y: number; layer: number; index: number; activation: number }[] = []
    const connections: { from: number; to: number; weight: number }[] = []
    const signals: { x: number; y: number; targetX: number; targetY: number; progress: number; value: number }[] = []

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
            activation: l === 0 ? Math.random() : 0, // Random activation for input layer
          })
        }
      }

      // Create connections
      for (let l = 0; l < layers.length - 1; l++) {
        const startNeurons = neurons.filter((n) => n.layer === l)
        const endNeurons = neurons.filter((n) => n.layer === l + 1)

        for (const start of startNeurons) {
          for (const end of endNeurons) {
            connections.push({
              from: start.index,
              to: end.index,
              weight: Math.random() * 2 - 1, // Random weight between -1 and 1
            })
          }
        }
      }
    }

    // Sigmoid activation function
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

    // Propagate signal through the network
    const propagateSignal = () => {
      // Reset activations for non-input layers
      for (const neuron of neurons) {
        if (neuron.layer > 0) {
          neuron.activation = 0
        }
      }

      // Propagate layer by layer
      for (let l = 0; l < layers.length - 1; l++) {
        const currentNeurons = neurons.filter((n) => n.layer === l)
        const nextNeurons = neurons.filter((n) => n.layer === l + 1)

        // For each neuron in the current layer
        for (const current of currentNeurons) {
          // Find all connections from this neuron
          const outgoingConnections = connections.filter((c) => c.from === current.index)

          // For each connection
          for (const conn of outgoingConnections) {
            const targetNeuron = neurons.find((n) => n.index === conn.to)
            if (targetNeuron) {
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
        }

        // Apply activation function to next layer
        for (const next of nextNeurons) {
          next.activation = sigmoid(next.activation)
        }
      }
    }

    // Animation variables
    let time = 0
    let lastPropagationTime = 0

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update time
      time += 0.01

      // Occasionally update input layer and propagate
      if (time - lastPropagationTime > 2) {
        // Update input layer activations
        for (const neuron of neurons) {
          if (neuron.layer === 0) {
            neuron.activation = Math.random()
          }
        }

        // Propagate signal
        propagateSignal()

        lastPropagationTime = time
      }

      // Draw connections
      for (const conn of connections) {
        const start = neurons.find((n) => n.index === conn.from)
        const end = neurons.find((n) => n.index === conn.to)

        if (start && end) {
          // Color based on weight
          const weightColor =
            conn.weight >= 0
              ? `rgba(16, 185, 129, ${Math.abs(conn.weight) * 0.5})` // Green for positive
              : `rgba(239, 68, 68, ${Math.abs(conn.weight) * 0.5})` // Red for negative

          ctx.beginPath()
          ctx.moveTo(start.x, start.y)
          ctx.lineTo(end.x, end.y)
          ctx.strokeStyle = weightColor
          ctx.lineWidth = 1
          ctx.stroke()
        }
      }

      // Update and draw signals
      for (let i = 0; i < signals.length; i++) {
        const signal = signals[i]

        // Update position
        signal.progress += 0.02
        const x = signal.x + (signal.targetX - signal.x) * signal.progress
        const y = signal.y + (signal.targetY - signal.y) * signal.progress

        // Draw signal
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)

        // Color based on value
        const signalColor =
          signal.value >= 0
            ? `rgba(16, 185, 129, ${Math.min(1, Math.abs(signal.value) + 0.2)})` // Green for positive
            : `rgba(239, 68, 68, ${Math.min(1, Math.abs(signal.value) + 0.2)})` // Red for negative

        ctx.fillStyle = signalColor
        ctx.fill()

        // Remove completed signals
        if (signal.progress >= 1) {
          signals.splice(i, 1)
          i--
        }
      }

      // Draw neurons
      for (const neuron of neurons) {
        ctx.beginPath()
        ctx.arc(neuron.x, neuron.y, 10, 0, Math.PI * 2)

        // Color based on activation
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
        ctx.font = "10px Arial"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(neuron.activation.toFixed(1), neuron.x, neuron.y)
      }

      // Draw layer labels
      ctx.fillStyle = "#1e293b"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"

      const padding = 50
      const layerSpacing = (canvas.width - padding * 2) / (layers.length - 1)

      ctx.fillText("Input Layer", padding, 10)

      for (let l = 1; l < layers.length - 1; l++) {
        ctx.fillText(`Hidden Layer ${l}`, padding + l * layerSpacing, 10)
      }

      ctx.fillText("Output Layer", padding + (layers.length - 1) * layerSpacing, 10)

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

