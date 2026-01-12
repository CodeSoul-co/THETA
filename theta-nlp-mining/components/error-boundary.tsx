"use client"

import React, { Component, ErrorInfo, ReactNode } from "react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { AlertCircle, RefreshCw } from "lucide-react"

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    }
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo)
    this.setState({
      error,
      errorInfo,
    })
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-background">
          <div className="max-w-2xl w-full space-y-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>出现错误</AlertTitle>
              <AlertDescription>
                <p className="mb-2">应用程序遇到了一个错误。请尝试刷新页面或联系支持团队。</p>
                {this.state.error && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-sm font-medium mb-1">
                      错误详情
                    </summary>
                    <pre className="text-xs bg-muted p-2 rounded mt-2 overflow-auto">
                      {this.state.error.toString()}
                      {this.state.errorInfo?.componentStack && (
                        <div className="mt-2">
                          {this.state.errorInfo.componentStack}
                        </div>
                      )}
                    </pre>
                  </details>
                )}
              </AlertDescription>
            </Alert>
            <div className="flex gap-2">
              <Button onClick={this.handleReset} variant="outline">
                <RefreshCw className="w-4 h-4 mr-2" />
                重试
              </Button>
              <Button
                onClick={() => window.location.reload()}
                variant="default"
              >
                刷新页面
              </Button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
