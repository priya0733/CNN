"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Download, Copy, Check } from "lucide-react"

interface CodeDownloaderProps {
  code: string
  filename: string
  language: string
}

export default function CodeDownloader({ code, filename, language }: CodeDownloaderProps) {
  const [copied, setCopied] = useState(false)

  const handleDownload = () => {
    const element = document.createElement("a")
    const file = new Blob([code], { type: "text/plain" })
    element.href = URL.createObjectURL(file)
    element.download = filename
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative">
      <div className="absolute right-2 top-2 flex space-x-2">
        <Button
          variant="outline"
          size="icon"
          onClick={handleCopy}
          className="h-8 w-8 bg-background/80 backdrop-blur-sm"
        >
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          <span className="sr-only">Copy code</span>
        </Button>
        <Button
          variant="outline"
          size="icon"
          onClick={handleDownload}
          className="h-8 w-8 bg-background/80 backdrop-blur-sm"
        >
          <Download className="h-4 w-4" />
          <span className="sr-only">Download code</span>
        </Button>
      </div>
      <pre className="rounded-lg bg-muted p-4 overflow-x-auto">
        <code className={`language-${language}`}>{code}</code>
      </pre>
    </div>
  )
}

