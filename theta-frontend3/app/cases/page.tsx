"use client"

import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ArrowLeft, ExternalLink, FileText, Calendar, User, Bookmark } from "lucide-react"
import Link from "next/link"

/** 案例库：学术论文案例 */
const RESEARCH_CASES = [
  {
    id: 1,
    title: "A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts",
    author: "Egger R, Yu J",
    year: "2022",
    journal: "Frontiers in Sociology",
    link: "https://www.frontiersin.org/journals/sociology/articles/10.3389/fsoc.2022.886498/full",
    tags: ["LDA", "BERTopic", "Twitter"],
    category: "方法对比",
  },
  {
    id: 2,
    title: "Computational vs. qualitative: Analyzing different approaches in identifying networked frames",
    author: "Kermani H, et al.",
    year: "2023",
    journal: "International Journal of Social Research Methodology",
    link: "https://www.tandfonline.com/doi/full/10.1080/13645579.2023.2186566",
    tags: ["网络框架", "定性分析"],
    category: "方法论",
  },
  {
    id: 3,
    title: "AutoTM 2.0: Automatic Topic Modeling Framework for Documents Analysis",
    author: "Khodorchenko M, et al.",
    year: "2024",
    journal: "arXiv",
    link: "https://arxiv.org/abs/2410.00655",
    tags: ["自动化", "文档分析"],
    category: "工具框架",
  },
  {
    id: 4,
    title: "Prompting Large Language Models for Topic Modeling",
    author: "Wang H, et al.",
    year: "2023",
    journal: "IEEE",
    link: "https://ieeexplore.ieee.org/abstract/document/10386113",
    tags: ["LLM", "提示工程"],
    category: "深度学习",
  },
  {
    id: 5,
    title: "Enhancing Short-Text Topic Modeling with LLM-Driven Context Expansion",
    author: "Akash P S, et al.",
    year: "2024",
    journal: "arXiv",
    link: "https://arxiv.org/abs/2410.03071",
    tags: ["短文本", "上下文扩展"],
    category: "深度学习",
  },
  {
    id: 6,
    title: "Topic research in fuzzy domain: Based on LDA topic Modelling",
    author: "Yu D, et al.",
    year: "2023",
    journal: "Information Sciences",
    link: "https://www.sciencedirect.com/science/article/pii/S0020025523011854",
    tags: ["模糊领域", "LDA"],
    category: "应用研究",
  },
  {
    id: 7,
    title: "GOOD AND BAD SOCIOLOGY: DOES TOPIC MODELLING MAKE A DIFFERENCE?",
    author: "Baranowski M, et al.",
    year: "2021",
    journal: "Society Register",
    link: "https://pressto.amu.edu.pl/index.php/sr/article/view/31045",
    tags: ["社会学", "方法论"],
    category: "方法论",
  },
  {
    id: 8,
    title: "Exploring Trends in Environmental, Social, and Governance Themes",
    author: "Park J, et al.",
    year: "2022",
    journal: "Frontiers in Psychology",
    link: "https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.890435/full",
    tags: ["ESG", "情感分析"],
    category: "应用研究",
  },
]

export default function CasesPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50">
      {/* 页面背景装饰 */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-20 -left-20 w-80 h-80 bg-blue-200/30 rounded-full blur-3xl" />
        <div className="absolute top-40 -right-20 w-96 h-96 bg-purple-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-1/3 w-72 h-72 bg-indigo-200/20 rounded-full blur-3xl" />
      </div>

      {/* 顶部导航 */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/60">
        <div className="max-w-7xl mx-auto px-5 sm:px-6 h-14 flex items-center justify-between">
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-2 text-slate-600 hover:text-slate-900">
              <ArrowLeft className="w-4 h-4" />
              返回首页
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <img src="/theta-logo.png" alt="THETA" className="h-7 w-auto" />
          </div>
        </div>
      </header>

      {/* 页面标题 */}
      <section className="max-w-7xl mx-auto px-5 sm:px-6 pt-12 pb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-blue-50 text-blue-600 rounded-full text-sm font-medium mb-6">
            <Bookmark className="w-4 h-4" />
            学术资源
          </div>
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-black text-slate-900 tracking-tight mb-4">
            学术<span className="text-blue-600">案例库</span>
          </h1>
          <p className="text-slate-600 text-lg max-w-2xl mx-auto">
            探索使用主题模型进行研究的优秀学术论文，获取灵感和方法论参考
          </p>
        </motion.div>
      </section>

      {/* 案例网格 */}
      <section className="max-w-7xl mx-auto px-5 sm:px-6 pb-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {RESEARCH_CASES.map((paper, index) => (
            <motion.a
              key={paper.id}
              href={paper.link}
              target="_blank"
              rel="noopener noreferrer"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 + index * 0.05 }}
              className="group"
            >
              <Card className="h-full border-2 border-slate-200/90 bg-white hover:border-blue-300 hover:shadow-xl hover:shadow-blue-100/50 hover:-translate-y-2 transition-all duration-300 overflow-hidden">
                {/* 顶部装饰条 */}
                <div className="h-1 bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400" />
                
                <div className="p-5 flex flex-col h-full">
                  {/* 分类标签和年份 */}
                  <div className="flex items-center justify-between mb-3">
                    <span className="px-2 py-0.5 bg-blue-50 text-blue-600 text-xs font-semibold rounded-md">
                      {paper.year}
                    </span>
                    <ExternalLink className="w-4 h-4 text-slate-400 group-hover:text-blue-600 transition-colors" />
                  </div>

                  {/* 标题 */}
                  <h3 className="text-sm font-bold text-slate-900 mb-3 line-clamp-3 group-hover:text-blue-600 transition-colors leading-snug min-h-[60px]">
                    {paper.title}
                  </h3>

                  {/* 分类 */}
                  <div className="mb-3">
                    <span className="inline-flex items-center gap-1 text-xs text-indigo-600 font-medium">
                      <FileText className="w-3 h-3" />
                      {paper.category}
                    </span>
                  </div>

                  {/* 作者和期刊 */}
                  <div className="mt-auto space-y-2">
                    <div className="flex items-start gap-2 text-xs text-slate-600">
                      <User className="w-3 h-3 mt-0.5 shrink-0" />
                      <span className="line-clamp-1">{paper.author}</span>
                    </div>
                    <p className="text-xs text-slate-500 italic line-clamp-1 pl-5">
                      {paper.journal}
                    </p>

                    {/* 标签 */}
                    <div className="flex flex-wrap gap-1.5 pt-3 border-t border-slate-100">
                      {paper.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-0.5 bg-slate-100 text-slate-600 text-[10px] rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            </motion.a>
          ))}
        </motion.div>

        {/* 底部提示 */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-12 text-center"
        >
          <p className="text-sm text-slate-400 mb-4">
            更多案例持续更新中...
          </p>
          <Link href="/">
            <Button variant="outline" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              返回首页
            </Button>
          </Link>
        </motion.div>
      </section>
    </div>
  )
}
