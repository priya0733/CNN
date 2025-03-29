"use client"

import { useEffect, useRef, useState } from "react"

export default function FrameworkComparison() {
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

    // Comparison data
    const comparisonPoints = [
      { category: "Ease of Use", tensorflow: 0.7, pytorch: 0.9 },
      { category: "Deployment", tensorflow: 0.9, pytorch: 0.7 },
      { category: "Research", tensorflow: 0.8, pytorch: 0.9 },
      { category: "Production", tensorflow: 0.9, pytorch: 0.7 },
      { category: "Community", tensorflow: 0.8, pytorch: 0.8 },
      { category: "Debugging", tensorflow: 0.6, pytorch: 0.9 },
    ]

    // Animation variables
    let time = 0
    let currentPoint = 0
    let animationProgress = 0

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update time
      time += 0.01

      // Update animation progress
      if (time % 3 < 0.01) {
        currentPoint = (currentPoint + 1) % comparisonPoints.length
        animationProgress = 0
      }

      animationProgress = Math.min(1, animationProgress + 0.02)

      // Draw framework logos
      drawTensorFlowLogo(ctx, canvas.width / 4 - 50, 40, 30)
      drawPyTorchLogo(ctx, (canvas.width * 3) / 4 - 50, 40, 30)

      // Draw comparison title
      ctx.fillStyle = "#1e293b"
      ctx.font = "bold 18px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("Framework Comparison", canvas.width / 2, 30)

      // Draw current comparison point
      const point = comparisonPoints[currentPoint]

      ctx.font = "bold 16px Arial"
      ctx.fillText(point.category, canvas.width / 2, 80)

      // Draw comparison bars
      const barWidth = 100
      const barHeight = 20
      const tfX = canvas.width / 4 - barWidth / 2
      const pyX = (canvas.width * 3) / 4 - barWidth / 2
      const barY = 120

      // TensorFlow bar
      ctx.fillStyle = "#e2e8f0"
      ctx.fillRect(tfX, barY, barWidth, barHeight)

      ctx.fillStyle = "#ff6f00" // TensorFlow orange
      ctx.fillRect(tfX, barY, barWidth * point.tensorflow * animationProgress, barHeight)

      // PyTorch bar
      ctx.fillStyle = "#e2e8f0"
      ctx.fillRect(pyX, barY, barWidth, barHeight)

      ctx.fillStyle = "#ee4c2c" // PyTorch red
      ctx.fillRect(pyX, barY, barWidth * point.pytorch * animationProgress, barHeight)

      // Draw scores
      ctx.fillStyle = "#1e293b"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.fillText((point.tensorflow * 10).toFixed(1) + "/10", tfX + barWidth / 2, barY + barHeight + 20)
      ctx.fillText((point.pytorch * 10).toFixed(1) + "/10", pyX + barWidth / 2, barY + barHeight + 20)

      // Draw comparison table
      drawComparisonTable(ctx, canvas.width / 2, 180, canvas.width - 100, canvas.height - 200)

      // Continue animation
      const frame = requestAnimationFrame(animate)
      setAnimationFrame(frame)
    }

    // Draw TensorFlow logo
    const drawTensorFlowLogo = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number) => {
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

    // Draw PyTorch logo
    const drawPyTorchLogo = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number) => {
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

    // Draw comparison table
    const drawComparisonTable = (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      width: number,
      height: number,
    ) => {
      const rows = [
        ["Feature", "TensorFlow", "PyTorch"],
        ["Graph Type", "Static (Eager mode available)", "Dynamic (Define-by-run)"],
        ["Debugging", "More complex", "Intuitive (Python-native)"],
        ["Deployment", "TF Serving, TF Lite, TF.js", "TorchServe, TorchScript"],
        ["Community", "Large, industry-focused", "Growing, research-focused"],
        ["Learning Curve", "Steeper", "More Pythonic"],
      ]

      const rowHeight = height / rows.length
      const colWidth = width / 3

      // Draw table
      ctx.strokeStyle = "#cbd5e1"
      ctx.lineWidth = 1

      // Draw horizontal lines
      for (let i = 0; i <= rows.length; i++) {
        ctx.beginPath()
        ctx.moveTo(x - width / 2, y + i * rowHeight)
        ctx.lineTo(x + width / 2, y + i * rowHeight)
        ctx.stroke()
      }

      // Draw vertical lines
      for (let i = 0; i <= 3; i++) {
        ctx.beginPath()
        ctx.moveTo(x - width / 2 + i * colWidth, y)
        ctx.lineTo(x - width / 2 + i * colWidth, y + height)
        ctx.stroke()
      }

      // Fill header row
      ctx.fillStyle = "#f1f5f9"
      ctx.fillRect(x - width / 2, y, width, rowHeight)

      // Draw text
      ctx.fillStyle = "#1e293b"
      ctx.font = "bold 14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"

      // Draw header row
      for (let j = 0; j < 3; j++) {
        ctx.fillText(rows[0][j], x - width / 2 + j * colWidth + colWidth / 2, y + rowHeight / 2)
      }

      // Draw data rows
      ctx.font = "12px Arial"
      for (let i = 1; i < rows.length; i++) {
        for (let j = 0; j < 3; j++) {
          ctx.fillText(rows[i][j], x - width / 2 + j * colWidth + colWidth / 2, y + i * rowHeight + rowHeight / 2)
        }
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

