"use client"

import { useEffect, useRef, useState } from "react"

export default function DeepLearningApplications() {
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

    // Application categories
    const applications = [
      {
        name: "Computer Vision",
        color: "#3b82f6",
        examples: ["Image Recognition", "Object Detection", "Facial Recognition"],
      },
      { name: "Natural Language", color: "#ef4444", examples: ["Translation", "Sentiment Analysis", "Chatbots"] },
      { name: "Speech", color: "#10b981", examples: ["Voice Assistants", "Transcription"] },
      { name: "Healthcare", color: "#f59e0b", examples: ["Disease Diagnosis", "Drug Discovery"] },
      { name: "Autonomous Vehicles", color: "#8b5cf6", examples: ["Self-driving Cars", "Drones"] },
    ]

    // Animation variables
    let time = 0
    let currentCategory = 0
    let currentExample = 0
    let fadeIn = 0

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update time
      time += 0.01
      fadeIn = Math.min(1, fadeIn + 0.02)

      // Change example every 3 seconds
      if (time % 3 < 0.01) {
        fadeIn = 0
        currentExample = (currentExample + 1) % applications[currentCategory].examples.length

        // Change category when we've gone through all examples
        if (currentExample === 0) {
          currentCategory = (currentCategory + 1) % applications.length
        }
      }

      // Draw neural network background
      drawNetworkBackground(ctx, canvas.width, canvas.height)

      // Draw current application
      const app = applications[currentCategory]
      const example = app.examples[currentExample]

      // Draw category
      ctx.fillStyle = app.color
      ctx.font = "bold 24px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.globalAlpha = fadeIn
      ctx.fillText(app.name, canvas.width / 2, canvas.height / 2 - 20)

      // Draw example
      ctx.fillStyle = "#1e293b"
      ctx.font = "18px Arial"
      ctx.fillText(example, canvas.width / 2, canvas.height / 2 + 20)
      ctx.globalAlpha = 1

      // Continue animation
      const frame = requestAnimationFrame(animate)
      setAnimationFrame(frame)
    }

    // Draw neural network background
    const drawNetworkBackground = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      const nodes: { x: number; y: number; size: number }[] = []
      const numNodes = 20

      // Create nodes if they don't exist
      if (nodes.length === 0) {
        for (let i = 0; i < numNodes; i++) {
          nodes.push({
            x: Math.random() * width,
            y: Math.random() * height,
            size: Math.random() * 3 + 2,
          })
        }
      }

      // Draw connections
      ctx.strokeStyle = "rgba(203, 213, 225, 0.3)"
      ctx.lineWidth = 1

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance < 100) {
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.stroke()
          }
        }
      }

      // Draw nodes
      ctx.fillStyle = "rgba(203, 213, 225, 0.5)"

      for (const node of nodes) {
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.size, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Start animation
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

