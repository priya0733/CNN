"use client"

import { useEffect, useRef, useState } from "react"

export default function NeuronAnimation() {
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

    // Neuron parameters
    const neuronX = canvas.width / 2
    const neuronY = canvas.height / 2
    const neuronRadius = 30

    // Input parameters
    const numInputs = 5
    const inputs: { x: number; y: number; value: number; weight: number }[] = []

    // Initialize inputs
    const initializeInputs = () => {
      inputs.length = 0

      const radius = Math.min(canvas.width, canvas.height) * 0.4

      for (let i = 0; i < numInputs; i++) {
        const angle = (i / numInputs) * Math.PI * 2
        inputs.push({
          x: neuronX + Math.cos(angle) * radius,
          y: neuronY + Math.sin(angle) * radius,
          value: Math.random(),
          weight: Math.random() * 2 - 1, // Random weight between -1 and 1
        })
      }
    }

    // Sigmoid activation function
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

    // Animation variables
    let time = 0
    let neuronActivation = 0
    let signalProgress = 0
    let showingActivation = false

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update time
      time += 0.01

      // Draw connections
      for (const input of inputs) {
        // Calculate input value (oscillating)
        input.value = (Math.sin(time * 2 + inputs.indexOf(input)) + 1) / 2

        // Draw connection
        ctx.beginPath()
        ctx.moveTo(input.x, input.y)
        ctx.lineTo(neuronX, neuronY)

        // Color based on weight
        const weightColor =
          input.weight >= 0
            ? `rgba(16, 185, 129, ${Math.abs(input.weight)})` // Green for positive
            : `rgba(239, 68, 68, ${Math.abs(input.weight)})` // Red for negative

        ctx.strokeStyle = weightColor
        ctx.lineWidth = 2
        ctx.stroke()

        // Draw input node
        ctx.beginPath()
        ctx.arc(input.x, input.y, 15, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(59, 130, 246, ${input.value})`
        ctx.fill()
        ctx.strokeStyle = "#1e293b"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw input value
        ctx.fillStyle = "#1e293b"
        ctx.font = "12px Arial"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(input.value.toFixed(2), input.x, input.y)
      }

      // Calculate weighted sum
      let weightedSum = 0
      for (const input of inputs) {
        weightedSum += input.value * input.weight
      }

      // Apply activation function
      neuronActivation = sigmoid(weightedSum)

      // Draw neuron
      ctx.beginPath()
      ctx.arc(neuronX, neuronY, neuronRadius, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(59, 130, 246, ${neuronActivation})`
      ctx.fill()
      ctx.strokeStyle = "#1e293b"
      ctx.lineWidth = 2
      ctx.stroke()

      // Draw activation value
      ctx.fillStyle = "#ffffff"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(neuronActivation.toFixed(2), neuronX, neuronY)

      // Draw signals
      if (!showingActivation) {
        signalProgress += 0.02

        for (const input of inputs) {
          const x = input.x + (neuronX - input.x) * signalProgress
          const y = input.y + (neuronY - input.y) * signalProgress

          if (signalProgress <= 1) {
            ctx.beginPath()
            ctx.arc(x, y, 5, 0, Math.PI * 2)
            ctx.fillStyle = input.weight >= 0 ? "#10b981" : "#ef4444"
            ctx.fill()
          }
        }

        if (signalProgress >= 1) {
          showingActivation = true

          // Draw output signal
          const outputX = neuronX + (canvas.width - neuronX) * 0.3
          const outputY = neuronY

          ctx.beginPath()
          ctx.moveTo(neuronX + neuronRadius, neuronY)
          ctx.lineTo(outputX, outputY)
          ctx.strokeStyle = "#3b82f6"
          ctx.lineWidth = 2
          ctx.stroke()

          // Draw output value
          ctx.beginPath()
          ctx.arc(outputX, outputY, 15, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(59, 130, 246, ${neuronActivation})`
          ctx.fill()
          ctx.strokeStyle = "#1e293b"
          ctx.lineWidth = 1
          ctx.stroke()

          ctx.fillStyle = "#1e293b"
          ctx.font = "12px Arial"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(neuronActivation.toFixed(2), outputX, outputY)

          // Reset after a delay
          setTimeout(() => {
            signalProgress = 0
            showingActivation = false

            // Randomize weights occasionally
            if (Math.random() < 0.3) {
              for (const input of inputs) {
                input.weight = Math.random() * 2 - 1
              }
            }
          }, 2000)
        }
      }

      // Continue animation
      const frame = requestAnimationFrame(animate)
      setAnimationFrame(frame)
    }

    // Initialize and start animation
    initializeInputs()
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

