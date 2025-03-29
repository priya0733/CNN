import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import NeuralNetworkAnimation from "@/components/animations/neural-network-animation"

export default function Home() {
  return (
    <div className="container mx-auto space-y-8">
      <div className="text-center space-y-4 py-10">
        <h1 className="text-4xl font-bold tracking-tight">Interactive Deep Learning Tutorial</h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Learn deep learning concepts with interactive visualizations and hands-on examples
        </p>
      </div>

      <div className="flex justify-center py-8">
        <div className="w-full max-w-3xl h-64 border rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
          <NeuralNetworkAnimation />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Introduction to Deep Learning</CardTitle>
            <CardDescription>Learn the fundamentals of neural networks and deep learning</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Explore the basic concepts, history, and applications of deep learning.</p>
          </CardContent>
          <CardFooter>
            <Link href="/introduction" className="w-full">
              <Button className="w-full">Start Learning</Button>
            </Link>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>TensorFlow & PyTorch Basics</CardTitle>
            <CardDescription>Get started with the most popular deep learning frameworks</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Compare TensorFlow and PyTorch, and learn the basic operations in both frameworks.</p>
          </CardContent>
          <CardFooter>
            <Link href="/frameworks" className="w-full">
              <Button className="w-full">Explore Frameworks</Button>
            </Link>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Activation & Loss Functions</CardTitle>
            <CardDescription>Understand the mathematical building blocks of neural networks</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Interactive visualizations of activation functions, loss functions, and backpropagation.</p>
          </CardContent>
          <CardFooter>
            <Link href="/functions" className="w-full">
              <Button className="w-full">Visualize Functions</Button>
            </Link>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Build a Simple Neural Network</CardTitle>
            <CardDescription>Hands-on project to implement your own neural network</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Step-by-step guide to build, train, and evaluate a neural network from scratch.</p>
          </CardContent>
          <CardFooter>
            <Link href="/hands-on" className="w-full">
              <Button className="w-full">Start Building</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}

