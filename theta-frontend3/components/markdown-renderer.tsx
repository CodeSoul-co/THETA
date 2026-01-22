"use client"

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { cn } from '@/lib/utils'

interface MarkdownRendererProps {
  content: string
  className?: string
}

export function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  return (
    <div className={cn("prose prose-sm max-w-none dark:prose-invert", className)}>
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        // 自定义标题样式
        h1: ({ children }) => (
          <h1 className="text-lg font-bold mt-4 mb-2 first:mt-0">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-base font-bold mt-3 mb-2 first:mt-0">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-sm font-bold mt-2 mb-1 first:mt-0">{children}</h3>
        ),
        // 自定义段落样式
        p: ({ children }) => (
          <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
        ),
        // 自定义列表样式
        ul: ({ children }) => (
          <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="leading-relaxed">{children}</li>
        ),
        // 自定义代码块样式
        code: ({ className, children, ...props }) => {
          const isInline = !className
          if (isInline) {
            return (
              <code 
                className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-sm font-mono text-blue-600 dark:text-blue-400"
                {...props}
              >
                {children}
              </code>
            )
          }
          return (
            <code className={cn("block", className)} {...props}>
              {children}
            </code>
          )
        },
        pre: ({ children }) => (
          <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto text-sm my-2">
            {children}
          </pre>
        ),
        // 自定义链接样式
        a: ({ href, children }) => (
          <a 
            href={href} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 underline"
          >
            {children}
          </a>
        ),
        // 自定义引用块样式
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-blue-500 pl-4 italic my-2 text-slate-600 dark:text-slate-400">
            {children}
          </blockquote>
        ),
        // 自定义表格样式
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="min-w-full border-collapse border border-slate-300 dark:border-slate-600">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-slate-300 dark:border-slate-600 px-3 py-2 bg-slate-100 dark:bg-slate-800 font-semibold text-left">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-slate-300 dark:border-slate-600 px-3 py-2">
            {children}
          </td>
        ),
        // 自定义分割线样式
        hr: () => (
          <hr className="my-4 border-slate-300 dark:border-slate-600" />
        ),
        // 自定义强调样式
        strong: ({ children }) => (
          <strong className="font-semibold text-slate-900 dark:text-slate-100">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="italic">{children}</em>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
    </div>
  )
}

export default MarkdownRenderer
