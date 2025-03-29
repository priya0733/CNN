"use client"

import { useEffect, useRef, useState } from "react"

export default function PyTorchAnimation() {
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

    // PyTorch dynamic graph nodes
    const nodes: {
      x: number
      y: number
      width: number
      height: number
      name: string
      type: string
      active: boolean
    }[] = [
      { x: 100, y: 50, width: 120, height: 40, name: "Input Tensor", type: "input", active: true },
      { x: 300, y: 50, width: 120, height: 40, name: "Linear", type: "operation", active: false },
      { x: 500, y: 50, width: 120, height: 40, name: "ReLU", type: "operation", active: false },
      { x: 300, y: 150, width: 120, height: 40, name: "Linear", type: "operation", active: false },
      { x: 500, y: 150, width: 120, height: 40, name: "Softmax", type: "operation", active: false },
      { x: 700, y: 100, width: 120, height: 40, name: "Output Tensor", type: "output", active: false },
    ]

    // Connections between nodes
    const connections: { from: number; to: number; active: boolean }[] = [
      { from: 0, to: 1, active: false },
      { from: 1, to: 2, active: false },
      { from: 2, to: 3, active: false },
      { from: 3, to: 4, active: false },
      { from: 4, to: 5, active: false },
    ]

    // Animation variables
    let time = 0
    let currentActiveNode = 0

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update time
      time += 0.01

      // Update active nodes and connections
      if (time % 1 < 0.02) {
        // Activate next node and connection
        if (currentActiveNode < nodes.length - 1) {
          connections[currentActiveNode].active = true
          currentActiveNode++
          nodes[currentActiveNode].active = true
        } else {
          // Reset animation after a delay
          setTimeout(() => {
            for (const node of nodes) {
              node.active = false
            }
            for (const conn of connections) {
              conn.active = false
            }
            currentActiveNode = 0
            nodes[currentActiveNode].active = true
          }, 1000)
        }
      }

      // Draw connections
      for (const conn of connections) {
        const start = nodes[conn.from]
        const end = nodes[conn.to]

        ctx.beginPath()
        ctx.moveTo(start.x + start.width, start.y + start.height / 2)
        ctx.lineTo(end.x, end.y + end.height / 2)

        if (conn.active) {
          ctx.strokeStyle = "#f59e0b" // Orange for active connections
          ctx.lineWidth = 3
        } else {
          ctx.strokeStyle = "#cbd5e1" // Gray for inactive connections
          ctx.lineWidth = 2
        }

        ctx.stroke()
      }

      // Draw nodes
      for (const node of nodes) {
        // Node background
        ctx.beginPath()
        ctx.roundRect(node.x, node.y, node.width, node.height, 8)

        if (node.active) {
          if (node.type === "input") {
            ctx.fillStyle = "#10b981" // Green for input
          } else if (node.type === "output") {
            ctx.fillStyle = "#ef4444" // Red for output
          } else {
            ctx.fillStyle = "#3b82f6" // Blue for operations
          }
        } else {
          ctx.fillStyle = "#94a3b8" // Gray for inactive nodes
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

      // Draw PyTorch logo
      drawPyTorchLogo(ctx, 20, 20, 30)

      // Draw dynamic graph explanation
      ctx.fillStyle = "#1e293b"
      ctx.font = "16px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("PyTorch Dynamic Computational Graph", canvas.width / 2, canvas.height - 30)
      ctx.font = "14px Arial"
      ctx.fillText("Operations are executed as they're defined", canvas.width / 2, canvas.height - 10)

      // Continue animation
      const frame = requestAnimationFrame(animate)
      setAnimationFrame(frame)
    }

    // Draw PyTorch logo
    const drawPyTorchLogo = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number) => {
      // PyTorch logo is a stylized flame
      ctx.fillStyle = "#ee4c2c" // PyTorch red

      // Draw the flame
      ctx.beginPath()
      ctx.moveTo(x + size * 0.5, y)
      ctx.bezierCurveTo(x + size * 0.8, y + size * 0.3, x + size * 0.8, y + size * 0.6, x + size * 0.5, y + size * 0.8)
      ctx.bezierCurveTo(x + size * 0.2, y + size * 0.6, x + size * 0.2, y + size * 0.3, x + size * 0.5, y)
      ctx.fill()

      // Draw text
      ctx.fillStyle = "#1e293b"
      ctx.font = "bold 16px Arial"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ctx.fillText("PyTorch", x + size + 10, y + size / 2)
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

