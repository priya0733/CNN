"use client"

import type React from "react"

import { useState } from "react"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Download, ExternalLink, Github } from "lucide-react"

export default function Sidebar() {
  const pathname = usePathname()
  const [email, setEmail] = useState("")

  const handleSubscribe = (e: React.FormEvent) => {
    e.preventDefault()
    alert(`Thank you for subscribing with ${email}!`)
    setEmail("")
  }

  const getRelatedLinks = () => {
    if (pathname === "/introduction") {
      return [
        { name: "Neural Networks Explained", url: "https://www.youtube.com/watch?v=aircAruvnKk" },
        { name: "Deep Learning Book", url: "https://www.deeplearningbook.org/" },
        { name: "History of Deep Learning", url: "https://en.wikipedia.org/wiki/Deep_learning#History" },
      ]
    } else if (pathname === "/frameworks") {
      return [
        { name: "TensorFlow Documentation", url: "https://www.tensorflow.org/guide" },
        { name: "PyTorch Tutorials", url: "https://pytorch.org/tutorials/" },
        {
          name: "Framework Comparison",
          url: "https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b",
        },
      ]
    } else if (pathname === "/functions") {
      return [
        { name: "Activation Functions Guide", url: "https://en.wikipedia.org/wiki/Activation_function" },
        {
          name: "Loss Functions Explained",
          url: "https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23",
        },
        { name: "Backpropagation Tutorial", url: "https://www.3blue1brown.com/topics/neural-networks" },
      ]
    } else if (pathname === "/hands-on") {
      return [
        { name: "NumPy Documentation", url: "https://numpy.org/doc/stable/" },
        { name: "Neural Network from Scratch", url: "https://victorzhou.com/blog/intro-to-neural-networks/" },
        {
          name: "Gradient Descent Explained",
          url: "https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3",
        },
      ]
    } else {
      return [
        { name: "Deep Learning Specialization", url: "https://www.coursera.org/specializations/deep-learning" },
        { name: "Fast.ai Course", url: "https://www.fast.ai/" },
        { name: "Machine Learning Mastery", url: "https://machinelearningmastery.com/start-here/" },
      ]
    }
  }

  return (
    <div className="hidden lg:block w-80 border-l p-4 overflow-y-auto">
      <div className="space-y-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Resources</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {getRelatedLinks().map((link, index) => (
                <a
                  key={index}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between text-sm hover:underline"
                >
                  <span>{link.name}</span>
                  <ExternalLink className="h-4 w-4" />
                </a>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Download Materials</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Button variant="outline" className="w-full justify-between">
                <span>Slides & Notes</span>
                <Download className="h-4 w-4" />
              </Button>
              <Button variant="outline" className="w-full justify-between">
                <span>Code Examples</span>
                <Download className="h-4 w-4" />
              </Button>
              <Button variant="outline" className="w-full justify-between">
                <span>Cheat Sheets</span>
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Newsletter</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubscribe} className="space-y-2">
              <input
                type="email"
                placeholder="Your email"
                className="w-full p-2 text-sm border rounded"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
              <Button type="submit" className="w-full">
                Subscribe
              </Button>
            </form>
            <p className="text-xs text-muted-foreground mt-2">Get updates on new tutorials and resources</p>
          </CardContent>
        </Card>

        <div className="flex justify-center">
          <a
            href="https://github.com/yourusername/deep-learning-tutorial"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center text-sm text-muted-foreground hover:text-foreground"
          >
            <Github className="h-4 w-4 mr-2" />
            <span>View on GitHub</span>
          </a>
        </div>
      </div>
    </div>
  )
}

