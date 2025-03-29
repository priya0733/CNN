"use client"

import { useEffect, useRef, useState } from "react"

export default function TensorFlowAnimation() {
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

    // TensorFlow graph nodes
    const nodes: { x: number; y: number; width: number; height: number; name: string; type: string }[] = [
      { x: 100, y: 50, width: 120, height: 40, name: "Input", type: "input" },
      { x: 300, y: 50, width: 120, height: 40, name: "Dense Layer", type: "operation" },
      { x: 500, y: 50, width: 120, height: 40, name: "ReLU", type: "operation" },
      { x: 300, y: 150, width: 120, height: 40, name: "Dense Layer", type: "operation" },
      { x: 500, y: 150, width: 120, height: 40, name: "Softmax", type: "operation" },
      { x: 700, y: 100, width: 120, height: 40, name: "Output", type: "output" },
    ]

    // Connections between nodes
    const connections: { from: number; to: number }[] = [
      { from: 0, to: 1 },
      { from: 1, to: 2 },
      { from: 2, to: 3 },
      { from: 3, to: 4 },
      { from: 4, to: 5 },
    ]

    // Data flow signals
    const signals: { x: number; y: number; targetX: number; targetY: number; progress: number }[] = []

    // Create a new signal
    const createSignal = () => {
      // Start from input node
      const startNode = nodes[0]
      const endNode = nodes[1]

      signals.push({
        x: startNode.x + startNode.width,
        y: startNode.y + startNode.height / 2,
        targetX: endNode.x,
        targetY: endNode.y + endNode.height / 2,
        progress: 0,
      })
    }

    // Animation variables
    let time = 0

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update time
      time += 0.01

      // Draw connections
      ctx.strokeStyle = "#cbd5e1"
      ctx.lineWidth = 2

      for (const conn of connections) {
        const start = nodes[conn.from]
        const end = nodes[conn.to]

        ctx.beginPath()
        ctx.moveTo(start.x + start.width, start.y + start.height / 2)
        ctx.lineTo(end.x, end.y + end.height / 2)
        ctx.stroke()
      }

      // Draw nodes
      for (const node of nodes) {
        // Node background
        ctx.beginPath()
        ctx.roundRect(node.x, node.y, node.width, node.height, 8)

        if (node.type === "input") {
          ctx.fillStyle = "#10b981" // Green for input
        } else if (node.type === "output") {
          ctx.fillStyle = "#ef4444" // Red for output
        } else {
          ctx.fillStyle = "#3b82f6" // Blue for operations
        }

        ctx.fill()

        // Node border
        ctx.strokeStyle = "#1e293b"
        ctx.lineWidth = 1
        ctx.stroke()

        // Node text
        ctx.fillStyle = "#ffffff"
        ctx.font = "14px Arial"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(node.name, node.x + node.width / 2, node.y + node.height / 2)
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
        ctx.arc(x, y, 5, 0, Math.PI * 2)
        ctx.fillStyle = "#f59e0b" // Orange for signals
        ctx.fill()

        // Remove completed signals and create new ones
        if (signal.progress >= 1) {
          signals.splice(i, 1)
          i--

          // Find the target node
          const targetNodeIndex = connections.find(
            (c) =>
              c.to ===
              connections.find(
                (c) => c.from === connections.find((c) => c.to === connections.find((c) => c.from === 0)?.to)?.to,
              )?.to,
          )?.to

          if (targetNodeIndex !== undefined) {
            const startNode = nodes[targetNodeIndex]
            const endNode = nodes[targetNodeIndex + 1]

            if (endNode) {
              signals.push({
                x: startNode.x + startNode.width,
                y: startNode.y + startNode.height / 2,
                targetX: endNode.x,
                targetY: endNode.y + endNode.height / 2,
                progress: 0,
              })
            }
          }
        }
      }

      // Occasionally create new signals
      if (Math.random() < 0.02 && signals.length < 10) {
        createSignal()
      }

      // Draw TensorFlow logo
      drawTensorFlowLogo(ctx, 20, 20, 30)

      // Continue animation
      const frame = requestAnimationFrame(animate)
      setAnimationFrame(frame)
    }

    // Draw TensorFlow logo
    const drawTensorFlowLogo = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number) => {
      // TensorFlow logo is a stylized 'T' and 'F'
      ctx.fillStyle = "#ff6f00" // TensorFlow orange

      // Draw the 'T'
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.lineTo(x + size, y)
      ctx.lineTo(x + size, y + size * 0.2)
      ctx.lineTo(x + size * 0.6, y + size * 0.2)
      ctx.lineTo(x + size * 0.6, y + size)
      ctx.lineTo(x + size * 0.4, y + size)
      ctx.lineTo(x + size * 0.4, y + size * 0.2)
      ctx.lineTo(x, y + size * 0.2)
      ctx.closePath()
      ctx.fill()

      // Draw text
      ctx.fillStyle = "#1e293b"
      ctx.font = "bold 16px Arial"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ctx.fillText("TensorFlow", x + size + 10, y + size / 2)
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

