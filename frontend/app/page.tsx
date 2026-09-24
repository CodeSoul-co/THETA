"use client"

import { OPEN_SOURCE_EDITION } from '@/lib/edition'

import type React from "react"
import { toast } from "sonner"
import { useState, useEffect, useCallback, useRef } from "react"
import Image from "next/image"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import {
  Play,
  BrainCircuit,
  MessageSquare,
  CheckCircle2,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Shield,
  Globe,
  FileText,
  Minus,
  Plus,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useAuth } from "@/contexts/auth-context"
import { useCyclingTypewriter } from "@/hooks/use-cycling-typewriter"
import { ProductScreenshot } from "@/components/landing/product-screenshot"
import { ParticlesBg } from "@/components/particles-bg"
import { isThetaPasswordValid, normalizeThetaPasswordInput, THETA_PASSWORD_POLICY_MESSAGE } from "@/lib/auth/password-policy"

/** 首页标题下打字机循环展示的平台功能文案（每条不同，轮流出现） */
const HERO_TYPEWRITER_PHRASES = [
  "数据清洗与预处理",
  "主题模型 (ETM) 训练与评估",
  "智能对话与 AI 科研助手",
  "任务中心 · 异步训练与监控",
  "可视化与结果导出",
  "从上传到洞察的一站式分析",
  "多数据集管理与协作",
  "深度主题发现与词云展示",
]

/** 产品实拍：从数据导入到结果导出。 */
const HOW_IT_WORKS_STEPS = [
  { title: "导入数据，确认分析内容", text: "上传表格后选择正文与元数据列；文本、PDF 和 Word 文档直接读取正文。开始前先预览数据，确认本次分析的内容。", image: "data", alt: "数据预览与分析列选择" },
  { title: "选择模型，设置分析参数", text: "在手动模式中选择主题模型、主题数和训练参数。需要嵌入的模型可配置本地模型，或使用自己填写的云端接口。", image: "models", alt: "主题模型与训练参数配置" },
  { title: "查看结果，继续追问", text: "查看主题关键词、分布和可视化结果。将图表引用到右侧猫咪科学家，配置自己的模型 API 后继续咨询与解读。", image: "results", alt: "主题模型的真实可视化结果" },
  { title: "导出图表与研究产物", text: "下载当前模型的完整结果包，也可按需导出图表、绘图数据与训练产物，继续完成论文或研究报告。", image: "export", alt: "模型结果与文件导出" },
]

/** 场景化分析实验室：上三下二 */
const SCENARIO_LAB_ROW1 = [
  {
    title: "心理健康与精细情感图谱",
    titleEn: "Mental Health & Micro-Emotion Mapping",
    tags: ["精神疾病类型", "负面情感检测"],
    icon: BrainCircuit,
  },
  {
    title: "金融合规与客诉风险洞察",
    titleEn: "Financial Compliance & Risk Insights",
    tags: ["FCPB 投诉分析", "风险洞察"],
    icon: Shield,
  },
  {
    title: "数字内容安全与净化",
    titleEn: "Content Moderation & Digital Safety",
    tags: ["仇恨言论识别", "垃圾账户过滤"],
    icon: MessageSquare,
  },
]
const SCENARIO_LAB_ROW2 = [
  {
    title: "跨语言与多文化语义分析",
    titleEn: "Cross-Lingual & Multicultural Analysis",
    tags: ["多语言混合处理", "德语专业文本"],
    icon: Globe,
  },
  {
    title: "长文本宏观语义理解",
    titleEn: "Long-Context Macro Understanding",
    tags: ["政治演讲全篇解析", "长帖语义聚合"],
    icon: FileText,
  },
]

/** FAQ 问答 */
const FAQ_ITEMS = [
  {
    q: "Theta 到底是什么？不需要编程也能用吗？",
    a: "Theta 是一个专为社会科学研究设计的 AI 分析平台。我们致力于降低科研门槛，您完全不需要编程基础。通过直观的对话界面，即可完成从数据清洗、主题建模到深度文本挖掘的全流程工作，让您专注于研究思路本身。",
  },
  {
    q: "我的数据安全吗？会被用于训练 AI 吗？",
    a: "本地版将项目数据与结果保存在你的电脑上。使用本地模型时，计算在本机完成；使用云端嵌入或对话 API 时，相关内容会发送给你配置的服务，请按数据要求选择使用方式。",
  },
  {
    q: "生成图表可以直接用于论文发表吗？",
    a: "完全可以。Theta 生成的所有可视化图表（矢量图）均符合主流学术期刊的出版标准。此外，我们还会提供详细的算法引用说明，方便您在论文的方法论部分准确撰写引用来源。",
  },
  {
    q: "支持多大的文件？支持哪些语言？",
    a: "我们专为大规模数据设计。数据量：支持 500MB 的 CSV/Excel 表格处理，轻松应对海量数据。语言支持：深度支持中文、英文及中英混合文本分析，同时兼容德语、法语等 20+ 种主流语言的语义理解。",
  },
]

export default function LandingPage() {
  const router = useRouter()
  const { user, login, register, sendVerificationCode, verifyCode, resetPassword } = useAuth()
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [modalMode, setModalMode] = useState<"login" | "register" | "reset">("login")
  const [loginPhone, setLoginPhone] = useState("")
  const [loginPassword, setLoginPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [rememberMe, setRememberMe] = useState(false)
  const [loginError, setLoginError] = useState("")
  const [loginNotice, setLoginNotice] = useState("")
  const [isLoggingIn, setIsLoggingIn] = useState(false)
  // 注册表单状态
  const [regUsername, setRegUsername] = useState("")
  const [regPhone, setRegPhone] = useState("")
  const [regPassword, setRegPassword] = useState("")
  const [regConfirm, setRegConfirm] = useState("")
  const [regFullName, setRegFullName] = useState("")
  const [showRegPassword, setShowRegPassword] = useState(false)
  const [showRegConfirm, setShowRegConfirm] = useState(false)
  const [regError, setRegError] = useState("")
  const [regNotice, setRegNotice] = useState("")
  const [regFieldErrors, setRegFieldErrors] = useState<Record<string, string>>({})
  const [isRegistering, setIsRegistering] = useState(false)
  const [regSuccess, setRegSuccess] = useState(false)
  // 验证码相关
  const [regCode, setRegCode] = useState("")
  const [countdown, setCountdown] = useState(0)
  const [isSendingCode, setIsSendingCode] = useState(false)
  const [isVerifyingCode, setIsVerifyingCode] = useState(false)
  const [registerStep, setRegisterStep] = useState<1 | 2>(1)
  const [resetEmail, setResetEmail] = useState("")
  const [resetCode, setResetCode] = useState("")
  const [resetNewPassword, setResetNewPassword] = useState("")
  const [resetConfirm, setResetConfirm] = useState("")
  const [resetError, setResetError] = useState("")
  const [resetNotice, setResetNotice] = useState("")
  const [resetCountdown, setResetCountdown] = useState(0)
  const [isSendingResetCode, setIsSendingResetCode] = useState(false)
  const [isResettingPassword, setIsResettingPassword] = useState(false)
  const hadAuthenticatedSessionRef = useRef(false)

  // 倒计时 effect
  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000)
      return () => clearTimeout(timer)
    }
  }, [countdown])
  useEffect(() => {
    if (resetCountdown > 0) {
      const timer = setTimeout(() => setResetCountdown(resetCountdown - 1), 1000)
      return () => clearTimeout(timer)
    }
  }, [resetCountdown])
  const [howItWorksStep, setHowItWorksStep] = useState(0)
  const [faqOpenIndex, setFaqOpenIndex] = useState<number | null>(null)
  const [showCiteModal, setShowCiteModal] = useState(false)
  const mouseGlowRef = useRef<HTMLDivElement>(null)

  const handlePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const glow = mouseGlowRef.current
    if (!glow) return
    glow.style.left = `${event.clientX}px`
    glow.style.top = `${event.clientY}px`
    glow.style.opacity = "1"
  }, [])

  const handlePointerLeave = useCallback(() => {
    if (mouseGlowRef.current) mouseGlowRef.current.style.opacity = "0"
  }, [])

  const howItWorksCurrent = HOW_IT_WORKS_STEPS[howItWorksStep]

  const { displayedText: typewriterText } = useCyclingTypewriter({
    phrases: HERO_TYPEWRITER_PHRASES,
    typingSpeed: 90,
    deleteSpeed: 60,
    holdDuration: 1600,
    loop: true,
  })

  // Load remembered phone
  useEffect(() => {
    const rememberedPhone = localStorage.getItem('remembered_phone')
    if (rememberedPhone) {
      setLoginPhone(rememberedPhone)
      setRememberMe(true)
    }
  }, [])

  useEffect(() => {
    if (user) {
      hadAuthenticatedSessionRef.current = true
      return
    }
    if (!hadAuthenticatedSessionRef.current) return
    setLoginPhone("")
    setLoginPassword("")
    setRememberMe(false)
    setLoginError("")
    setLoginNotice("")
    localStorage.removeItem('remembered_phone')
  }, [user])

  useEffect(() => {
    if (OPEN_SOURCE_EDITION) return
    const requestedMode = new URLSearchParams(window.location.search).get('auth')
    if (requestedMode !== 'login' && requestedMode !== 'register') return
    setModalMode(requestedMode)
    setShowLoginModal(true)
    window.history.replaceState(null, '', window.location.pathname)
  }, [])

  // 已登录用户停留在起始页，不自动跳转到工作台。
  // 若出现「过几秒自动跳到项目管理界面」，多半是误触了上述入口，此处不做任何自动跳转

  const handleStartAnalysis = () => {
    if (!user && !OPEN_SOURCE_EDITION) {
      setModalMode("login")
      setShowLoginModal(true)
      return
    }
    router.push("/workbench")
  }

  const handleLogin = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    
    if (!/^1[3-9]\d{9}$/.test(loginPhone)) {
      setLoginError("请输入有效的中国大陆手机号")
      return
    }
    if (!loginPassword.trim()) {
      setLoginError("请输入密码")
      return
    }

    setIsLoggingIn(true)
    setLoginError("")
    setLoginNotice("")

    try {
      await login(`+86${loginPhone}`, loginPassword, rememberMe)
      
      // Save remembered phone if selected
      if (rememberMe) {
        localStorage.setItem('remembered_phone', loginPhone)
      } else {
        localStorage.removeItem('remembered_phone')
      }
      
      setShowLoginModal(false)
      setLoginPhone("")
      setLoginPassword("")
      router.push("/workbench")
    } catch (err: unknown) {
      let errorMessage = "登录失败，请检查手机号和密码"
      if (err instanceof Error) {
        errorMessage = err.message || errorMessage
        const isConnectionError =
          errorMessage.includes('连接') ||
          errorMessage.includes('超时') ||
          errorMessage.includes('fetch') ||
          errorMessage.includes('network') ||
          errorMessage.includes('Failed to fetch')
        if (isConnectionError) {
          errorMessage = '登录服务暂不可用，请检查本地前端服务配置。'
        } else if (errorMessage.includes('401') || errorMessage.includes('Unauthorized') || errorMessage.includes('手机号或密码错误') || errorMessage.includes('用户名或密码错误')) {
          errorMessage = '手机号或密码错误'
        }
      }
      setLoginError(errorMessage)
    } finally {
      setIsLoggingIn(false)
    }
  }

  const handleGoToRegister = () => {
    setModalMode("register")
    setRegisterStep(1)
    setLoginError("")
    setRegError("")
    setRegNotice("")
    setRegFieldErrors({})
    setRegSuccess(false)
  }

  const handleGoToLogin = () => {
    setModalMode("login")
    setLoginError("")
  }

  const handleGoToReset = () => {
    setModalMode("reset")
    setLoginNotice("")
    setResetEmail("")
    setResetError("")
    setResetNotice("")
  }

  const handleSendResetCode = async () => {
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(resetEmail) || resetCountdown > 0) {
      setResetError("请先输入有效的注册邮箱")
      return
    }
    setIsSendingResetCode(true)
    setResetError("")
    try {
      const result = await sendVerificationCode({ email: resetEmail, type: "reset_password" })
      setResetNotice(result.debug_code
        ? `${result.message}，调试验证码：${result.debug_code}`
        : "如果该邮箱已注册，验证码邮件将很快送达。")
      setResetCountdown(60)
    } catch (error: unknown) {
      setResetError(error instanceof Error ? error.message : "验证码发送失败，请稍后重试")
    } finally {
      setIsSendingResetCode(false)
    }
  }

  const handleResetPassword = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(resetEmail)) return setResetError("请输入有效的注册邮箱")
    if (!/^\d{6}$/.test(resetCode)) return setResetError("请输入 6 位验证码")
    if (!isThetaPasswordValid(resetNewPassword)) {
      return setResetError(THETA_PASSWORD_POLICY_MESSAGE)
    }
    if (resetNewPassword !== resetConfirm) return setResetError("两次输入的新密码不一致")
    setIsResettingPassword(true)
    setResetError("")
    try {
      await resetPassword({ email: resetEmail, code: resetCode, new_password: resetNewPassword })
      setLoginPhone("")
      setLoginPassword("")
      setResetEmail(""); setResetCode(""); setResetNewPassword(""); setResetConfirm("")
      setModalMode("login")
      setLoginNotice("密码已重置，请使用注册手机号和新密码登录。")
    } catch (error: unknown) {
      setResetError(error instanceof Error ? error.message : "密码重置失败，请稍后重试")
    } finally {
      setIsResettingPassword(false)
    }
  }

  const handleSendCode = async () => {
    // 验证邮箱格式
    if (!regPhone || !/^1[3-9]\d{9}$/.test(regPhone)) {
      setRegFieldErrors({ phone: "请先输入有效的中国大陆手机号" })
      return
    }
    if (countdown > 0) return

    setIsSendingCode(true)
    setRegFieldErrors({ ...regFieldErrors, code: "" })
    setRegNotice("")
    try {
      const result = await sendVerificationCode({
        phone: `+86${regPhone}`,
        type: "register"
      })
      setRegError("")
      setRegNotice(
        result.debug_code
          ? `${result.message || "验证码已生成"}，调试验证码：${result.debug_code}`
          : result.message || "验证码已发送，请查收手机短信"
      )
      setCountdown(60) // 60秒倒计时
    } catch (error: any) {
      setRegError(error.message || "发送验证码失败，请稍后重试")
    } finally {
      setIsSendingCode(false)
    }
  }

  const handleVerifyRegistrationCode = async () => {
    if (!/^1[3-9]\d{9}$/.test(regPhone)) {
      setRegFieldErrors({ phone: "请输入有效的中国大陆手机号" })
      return
    }
    if (!/^\d{6}$/.test(regCode)) {
      setRegFieldErrors({ code: "请输入 6 位验证码" })
      return
    }
    setIsVerifyingCode(true)
    setRegError("")
    setRegFieldErrors({})
    try {
      const valid = await verifyCode({ phone: `+86${regPhone}`, code: regCode, type: "register" })
      if (!valid) {
        setRegFieldErrors({ code: "验证码无效或已过期，请重新获取" })
        return
      }
      setRegNotice("验证码验证通过，请设置账号和密码")
      setRegisterStep(2)
    } catch (error: unknown) {
      setRegError(error instanceof Error ? error.message : "验证码校验失败，请稍后重试")
    } finally {
      setIsVerifyingCode(false)
    }
  }

  const handleRegister = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (registerStep !== 2) return
    const fieldErrs: Record<string, string> = {}
    if (!/^[\p{L}\p{N}_.-]{2,64}$/u.test(regUsername.trim())) fieldErrs.username = "用户名需为 2-64 位字母、数字、下划线、点或连字符"
    if (!regPhone || !/^1[3-9]\d{9}$/.test(regPhone)) fieldErrs.phone = "请输入有效的中国大陆手机号"
    if (!isThetaPasswordValid(regPassword)) fieldErrs.password = THETA_PASSWORD_POLICY_MESSAGE
    if (regPassword !== regConfirm) fieldErrs.confirm = "两次输入的密码不一致"
    if (!regCode || regCode.length !== 6) fieldErrs.code = "请输入 6 位验证码"
    if (Object.keys(fieldErrs).length > 0) { setRegFieldErrors(fieldErrs); return }
    setIsRegistering(true)
    setRegError("")
    setRegNotice("")
    setRegFieldErrors({})
    try {
      await register(regUsername, `+86${regPhone}`, regPassword, regFullName || undefined, regCode)
      setRegSuccess(true)
      // 注册完成后回到登录，避免自动建立并缓存一个未验证的会话。
      setTimeout(() => {
        setRegSuccess(false)
        setRegUsername(""); setRegPhone(""); setRegPassword(""); setRegConfirm(""); setRegFullName(""); setRegCode(""); setCountdown(0)
      setLoginPhone(regPhone)
        setRegisterStep(1)
        setModalMode("login")
      }, 1800)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      if (msg.includes("Username already registered") || msg.includes("用户名")) {
        setRegFieldErrors({ username: "该用户名已被注册" })
      } else if (msg.includes("Phone already registered") || msg.includes("手机号")) {
        setRegFieldErrors({ phone: "该手机号已被注册" })
      } else if (msg.includes("无法连接") || msg.includes("fetch") || msg.includes("network") || msg.includes("Failed to fetch")) {
        setRegError("无法连接到服务器，请检查后端是否已启动")
      } else {
        setRegError(msg || "注册失败，请稍后重试")
      }
    } finally {
      setIsRegistering(false)
    }
  }

  return (
    <div
      className="min-h-screen relative"
      onPointerMove={handlePointerMove}
      onPointerLeave={handlePointerLeave}
    >
      {/* 全页背景特效：渐变 + 点阵 + 柔光球 */}
      <div className="page-bg-effect" aria-hidden>
        <div className="page-bg-effect__gradient" />
        <div className="page-bg-effect__dots" />
        <div className="page-bg-effect__orb page-bg-effect__orb--1" />
        <div className="page-bg-effect__orb page-bg-effect__orb--2" />
        <div className="page-bg-effect__orb page-bg-effect__orb--3" />
        <div className="page-bg-effect__orb page-bg-effect__orb--4" />
        <div className="page-bg-effect__orb page-bg-effect__orb--5" />
      </div>
      {/* 粒子连线背景：点 + 线，鼠标靠近时连线并轻微吸附 */}
      <ParticlesBg zIndex={-1} opacity={0.32} color="59, 130, 246" count={48} />
      {/* 鼠标跟随光晕：独立层叠在内容之上，pointer-events: none 不挡点击 */}
      <div ref={mouseGlowRef} className="page-bg-effect__mouse-glow" aria-hidden />
      {/* Navigation Bar */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="sticky top-0 z-50 bg-white/98 backdrop-blur-md border-b border-slate-200/80 shadow-sm shadow-slate-900/5"
      >
        <div className="max-w-7xl mx-auto px-5 sm:px-6 h-14 sm:h-16 flex items-center justify-between">
          {/* Left: Logo - 使用上传的 THETA 图片（青绿+橙边） */}
          <div className="flex items-center">
            <Image unoptimized
              src="/theta-logo.png"
              alt="THETA"
              width={1653}
              height={938}
              priority
              sizes="(min-width: 640px) 64px, 56px"
              className="h-8 sm:h-9 w-auto object-contain"
            />
          </div>

          {/* Center: Navigation Links */}
          <nav className="hidden md:flex items-center gap-7">
            {[
              { label: "首页", href: "#" },
              { label: "关于THETA", href: "#core-features" },
              { label: "案例库", href: "/cases" },
              { label: "文档", href: "https://codesoul-co.github.io/THETA/zh/", external: true },
              { label: "团队成员", href: "/team" },
              { label: "帮助中心", href: "#faq" },
            ].map((link) => (
              <a
                key={link.label}
                href={link.href}
                {...("external" in link && link.external
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
                className="nav-link text-[13px] sm:text-sm"
              >
                {link.label}
              </a>
            ))}
          </nav>

          {/* Right: Open Source Links + Auth Buttons */}
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/CodeSoul-co/THETA"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link flex items-center gap-1.5 text-[13px] sm:text-sm"
              title="GitHub"
            >
              <img
                src="/github-mark.svg"
                alt="GitHub"
                width={18}
                height={18}
                className="h-4 w-4 sm:h-[18px] sm:w-[18px] object-contain shrink-0"
              />
              <span className="hidden sm:inline">GitHub</span>
            </a>
            <a
              href="https://huggingface.co/organizations/CodeSoulco"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link flex items-center gap-1.5 text-[13px] sm:text-sm"
              title="Hugging Face - CodeSoul"
              aria-label="打开 Hugging Face CodeSoul 组织页"
            >
              <img
                src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
                alt="Hugging Face"
                className="h-4 w-4 sm:h-[18px] sm:w-[18px] object-contain"
              />
              <span className="hidden sm:inline">Hugging Face</span>
            </a>
            <span className="w-px h-5 bg-slate-200" aria-hidden />
            {OPEN_SOURCE_EDITION ? (<Button onClick={handleStartAnalysis} className="bg-blue-600 text-white">进入工作台</Button>) : user ? (
              <>
                <span className="text-[13px] sm:text-sm text-slate-600 font-medium">欢迎, {user.username}</span>
                <Button 
                  onClick={() => router.push("/workbench")}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 sm:px-5 text-sm font-medium rounded-lg shadow-sm"
                >
                  进入工作台
                </Button>
              </>
            ) : (
              <>
                <Button 
                  variant="ghost" 
                  className="nav-link text-slate-700 hover:bg-slate-100/80 rounded-lg px-4"
                  onClick={() => setShowLoginModal(true)}
                >
                  登录
                </Button>
                <Button 
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 sm:px-5 text-sm font-medium rounded-lg shadow-sm hover:shadow-md transition-shadow"
                  onClick={handleStartAnalysis}
                >
                  免费使用
                </Button>
              </>
            )}
          </div>
        </div>
      </motion.header>

      {/* Hero Section - 填满视口，底部为跑马灯，一屏平铺 */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50/20 to-white pb-4 flex flex-col min-h-[calc(100vh-4rem)]">
        {/* Subtle data pattern background */}
        <div className="absolute inset-0 overflow-hidden">
          <svg className="absolute top-0 right-0 w-1/2 h-full opacity-[0.04]" viewBox="0 0 400 400">
            <defs>
              <pattern id="dots" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
                <circle cx="2" cy="2" r="1.5" fill="#3B82F6" />
              </pattern>
            </defs>
            <rect fill="url(#dots)" width="400" height="400" />
          </svg>
          <div className="absolute top-20 right-20 w-96 h-96 bg-blue-100/25 rounded-full blur-3xl" />
          <div className="absolute bottom-20 left-20 w-64 h-64 bg-indigo-100/15 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-7xl mx-auto px-5 sm:px-6 py-4 lg:py-6 flex-1 min-h-0 flex flex-col">
          <div className="grid grid-cols-1 lg:grid-cols-2 lg:grid-rows-[1fr] gap-6 lg:gap-8 items-stretch min-h-0 flex-1">
            {/* Left Column - 松散大气，主内容 + 跑马灯；min-w-0 防止内容撑开列宽 */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              className="flex flex-col justify-center min-w-0"
            >
              <h1 className="hero-title text-3xl sm:text-4xl md:text-5xl lg:text-[3rem] text-slate-900 mb-4 leading-[1.2] text-balance">
                零代码社科文本挖掘工作台
              </h1>
              <p className="hero-typewriter text-lg sm:text-xl md:text-2xl text-blue-600 mb-4 min-h-[2.25rem] sm:min-h-[2.5rem] flex items-center">
                <span>{typewriterText}</span>
                <span className="inline-block w-[3px] h-5 sm:h-6 bg-blue-500 ml-0.5 animate-pulse rounded-sm align-middle shrink-0" aria-hidden />
              </p>
              <p className="text-base sm:text-lg text-slate-600 mb-6 leading-[1.6] text-pretty max-w-xl">
                从数据清洗到深度洞察，AI 驱动的全流程科研助手。上传数据，对话式交互，即刻获取专业分析结果。
              </p>
              <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 items-start sm:items-center">
                <span className="hero-cta-marquee inline-block rounded-xl p-[2px]">
                  <Button
                    onClick={handleStartAnalysis}
                    size="lg"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-6 sm:px-8 py-3.5 sm:py-4 text-sm font-semibold rounded-[10px] shadow-lg shadow-blue-600/25 hover:shadow-xl hover:shadow-blue-600/30 transition-all w-full sm:w-auto"
                  >
                    立即开始分析
                  </Button>
                </span>
                <Button
                  variant="outline"
                  size="lg"
                  className="border-slate-200 hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50/50 px-6 sm:px-8 py-3.5 sm:py-4 text-sm font-medium rounded-xl bg-white/80"
                  onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}
                >
                  <Play className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
                  查看使用流程
                </Button>
              </div>
              <div className="flex items-center gap-8 sm:gap-10 mt-6 pt-5 border-t border-slate-200/80">
                {[
                  { value: "Web", label: "浏览器工作台" },
                  { value: "CLI", label: "命令行 Agent" },
                  { value: "桌面端", label: "macOS 与 Windows" },
                ].map((stat) => (
                  <div key={stat.label}>
                    <p className="stat-value text-lg sm:text-xl text-blue-600">{stat.value}</p>
                    <p className="stat-label text-xs sm:text-sm mt-0.5 text-slate-500">{stat.label}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6, delay: 0.2 }} className="min-w-0 self-center">
              <ProductScreenshot name="results" alt="THETA 手动工作台 · 城市公共服务反馈分析" priority />
              <p className="mt-4 text-center text-xs text-slate-500">真实产品界面 · 从主题发现到图表导出</p>
            </motion.div>
          </div>
        </div>

        {/* 跑马灯：页面底部，一打开即见 */}
        <div className="page-bg-effect__ghost-text page-bg-effect__ghost-text--hero mt-2 sm:mt-3 shrink-0">
          <div className="page-bg-effect__ghost-text-track">
            <svg className="page-bg-effect__ghost-text-svg" viewBox="0 0 1800 120" preserveAspectRatio="xMinYMid meet" aria-hidden>
              <text x="0" y="92" fill="none" stroke="rgba(148,163,184,0.5)" strokeWidth="2" fontSize="108" fontWeight="800">THETA · THETA · THETA · THETA · THETA · THETA · THETA ·</text>
            </svg>
            <svg className="page-bg-effect__ghost-text-svg" viewBox="0 0 1800 120" preserveAspectRatio="xMinYMid meet" aria-hidden>
              <text x="0" y="92" fill="none" stroke="rgba(148,163,184,0.5)" strokeWidth="2" fontSize="108" fontWeight="800">THETA · THETA · THETA · THETA · THETA · THETA · THETA ·</text>
            </svg>
          </div>
        </div>
      </section>

      {/* 二、核心功能引擎 - 三大横向模块（图文交替） */}
      <section id="core-features" className="max-w-7xl mx-auto px-5 sm:px-6 py-20 sm:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">核心功能引擎：重构社科研究生产力</h2>
          <p className="text-slate-600 max-w-2xl mx-auto text-[15px] sm:text-base leading-relaxed">
            集成大语言模型与主题模型，打造遵循严谨学术范式的一站式研究工作流。
          </p>
        </motion.div>

        {/* 模块一：全栈式主题建模 - 左文案 | 右实拍 */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center mb-16 sm:mb-20"
        >
          <div>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4">全栈式主题建模</h3>
            <p className="text-slate-600 leading-[1.7] mb-4">
              不再受限于单一模型。Theta 集成了从经典统计学到最新深度学习的完整算法库，灵活适配短文本、长文档及动态演化数据，满足多元化的研究目标。
            </p>
            <p className="text-xs text-slate-500">LDA · ETM · CTM · DTM · <strong className="text-blue-600">BERTopic</strong> · <strong className="text-blue-600">THETA</strong></p>
          </div>
          <div><ProductScreenshot name="models" alt="主题模型与参数配置" /></div>
        </motion.div>

        {/* 模块二：本地分析，进度可见 - 左实拍 | 右文案 */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center mb-16 sm:mb-20"
        >
          <div className="order-2 lg:order-1"><ProductScreenshot name="training" alt="训练执行日志与实时进度" /></div>
          <div className="order-1 lg:order-2">
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4">本地分析，进度可见</h3>
            <p className="text-slate-600 leading-[1.7] mb-4">
              桌面端内置 Python 运行环境，上传的数据在本地处理。训练轮次、嵌入批次和损失值随实际执行持续更新。
            </p>
            <ul className="space-y-2 text-slate-600">
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 桌面端内置运行环境</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 训练与嵌入进度实时可见</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 任务与结果保存在本机</li>
            </ul>
          </div>
        </motion.div>

        {/* 模块三：交互式 AI 科学家 - 左文案 | 右交互演示图 */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center"
        >
          <div>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4">交互式 AI 科学家</h3>
            <p className="text-slate-600 leading-[1.7] mb-4">
              在对话模式中讨论研究问题，或在手动模式中引用图表继续追问。填写自己的模型 API 后，即可围绕数据、方法和结果与猫咪科学家交流。
            </p>
            <ul className="space-y-2 text-slate-600">
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 非模板化深度解读</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 图表级多轮追问</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 研究假设辅助验证</li>
            </ul>
          </div>
          <div><ProductScreenshot name="assistant" alt="猫咪科学家对话工作台" /></div>
        </motion.div>
      </section>

      {/* 三、使用教程 How it Works - 左右布局 + 图文轮播 */}
      <section id="how-it-works" className="max-w-7xl mx-auto px-5 sm:px-6 py-20 sm:py-24 bg-slate-50/50">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">怎么用？</h2>
          <p className="text-slate-600 max-w-xl mx-auto text-[15px] sm:text-base leading-relaxed">
            零代码基础也能用，四步完成从数据到洞察
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-center"
        >
          {/* 左侧：当前步骤文案 */}
          <div className="order-2 lg:order-1">
            <p className="text-xs sm:text-sm text-blue-600 font-semibold tracking-wide mb-2">
              步骤 {howItWorksStep + 1} / {HOW_IT_WORKS_STEPS.length}
            </p>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4 tracking-tight">
              {HOW_IT_WORKS_STEPS[howItWorksStep].title}
            </h3>
            <p className="text-slate-600 leading-[1.7]">
              {HOW_IT_WORKS_STEPS[howItWorksStep].text}
            </p>
          </div>
          <div className="order-1 lg:order-2 min-w-0">
            <ProductScreenshot key={howItWorksCurrent.image} name={howItWorksCurrent.image} alt={howItWorksCurrent.alt} />
            <div className="mt-4 flex items-center justify-between gap-4">
              <button type="button" onClick={() => setHowItWorksStep(s => (s + 3) % 4)} className="rounded-full border border-slate-200 bg-white p-2 text-slate-600 hover:border-blue-300 hover:text-blue-600" aria-label="上一步"><ChevronLeft className="size-5" /></button>
              <div className="flex gap-2">{HOW_IT_WORKS_STEPS.map((step, index) => <button key={step.image} type="button" onClick={() => setHowItWorksStep(index)} aria-label={`第 ${index + 1} 步：${step.title}`} aria-pressed={howItWorksStep === index} className={`h-2.5 rounded-full transition-all ${howItWorksStep === index ? 'w-7 bg-blue-600' : 'w-2.5 bg-slate-300 hover:bg-blue-400'}`} />)}</div>
              <button type="button" onClick={() => setHowItWorksStep(s => (s + 1) % 4)} className="rounded-full border border-slate-200 bg-white p-2 text-slate-600 hover:border-blue-300 hover:text-blue-600" aria-label="下一步"><ChevronRight className="size-5" /></button>
            </div>
          </div>
        </motion.div>
      </section>

      {/* 四、场景化分析实验室 Scenario Lab - 上三下二 */}
      <section id="scenario-lab" className="max-w-7xl mx-auto px-5 sm:px-6 py-20 sm:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">场景化分析实验室</h2>
          <p className="text-slate-600 max-w-2xl mx-auto text-[15px] sm:text-base leading-relaxed">
            选择您的研究数据类型，即刻微调您的领域嵌入模型，发现更深入的领域洞察。
          </p>
        </motion.div>
        <div className="space-y-6">
          {/* 第一行：3 个卡片 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {SCENARIO_LAB_ROW1.map((card, index) => (
              <Card
                key={card.title}
                className="group relative border border-slate-200/90 bg-white hover:border-blue-200 hover:shadow-lg hover:shadow-blue-100/30 hover:-translate-y-1 transition-all p-6 rounded-2xl flex flex-col"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-slate-900 mb-1 tracking-tight">{card.title}</h3>
                    <p className="text-xs text-blue-600 font-medium mb-3">{card.titleEn}</p>
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {card.tags.map((tag) => (
                        <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity border-blue-200 text-blue-600 hover:bg-blue-50"
                      onClick={handleStartAnalysis}
                    >
                      立即分析
                    </Button>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0">
                    <card.icon className="w-6 h-6" />
                  </div>
                </div>
              </Card>
            ))}
          </motion.div>
          {/* 第二行：2 个宽卡片 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {SCENARIO_LAB_ROW2.map((card) => (
              <Card
                key={card.title}
                className="group relative border border-slate-200/90 bg-white hover:border-blue-200 hover:shadow-lg hover:shadow-blue-100/30 hover:-translate-y-1 transition-all p-6 rounded-2xl flex flex-col"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-slate-900 mb-1 tracking-tight">{card.title}</h3>
                    <p className="text-xs text-blue-600 font-medium mb-3">{card.titleEn}</p>
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {card.tags.map((tag) => (
                        <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity border-blue-200 text-blue-600 hover:bg-blue-50"
                      onClick={handleStartAnalysis}
                    >
                      Load Template
                    </Button>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0">
                    <card.icon className="w-6 h-6" />
                  </div>
                </div>
              </Card>
            ))}
          </motion.div>
        </div>
      </section>

      {/* 六、常见问题 FAQ */}
      <section id="faq" className="max-w-3xl mx-auto px-5 sm:px-6 py-20 sm:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">关于 Theta 的一切</h2>
          <p className="text-slate-600 text-[15px] sm:text-base leading-relaxed">
            我们整理了用户最关心的问题，助您无忧开启科研。
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="space-y-3"
        >
          {FAQ_ITEMS.map((item, index) => (
            <div
              key={index}
              className="border border-slate-200 rounded-xl bg-white overflow-hidden transition-[border-color,box-shadow] hover:border-blue-200 hover:shadow-md"
            >
              <button
                type="button"
                onClick={() => setFaqOpenIndex(faqOpenIndex === index ? null : index)}
                className="w-full flex items-center justify-between gap-4 px-5 py-4 text-left"
              >
                <span className="font-semibold text-slate-900">{item.q}</span>
                <span className="text-slate-400 shrink-0">
                  {faqOpenIndex === index ? <Minus className="w-5 h-5" /> : <Plus className="w-5 h-5" />}
                </span>
              </button>
              <div
                className={`grid transition-[grid-template-rows] duration-200 ease-in-out ${faqOpenIndex === index ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
              >
                <div className="overflow-hidden">
                  <p className="px-5 pb-4 pt-0 text-slate-600 text-sm leading-relaxed border-t border-slate-100">
                    {item.a}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </motion.div>
        <p className="text-center text-slate-500 text-sm mt-8">
          还有其他问题？直接联系我们的产品团队。
        </p>
        <div className="flex flex-wrap justify-center items-center gap-4 mt-6">
          <Button variant="outline" size="sm" className="h-9 px-4 shrink-0" asChild>
            <a href="mailto:duanzhenke@code-soul.com">留言给支持团队</a>
          </Button>
          <span className="text-slate-400 text-sm shrink-0">或</span>
          <a href="mailto:duanzhenke@code-soul.com" className="text-blue-600 text-sm font-medium hover:underline shrink-0">
            duanzhenke@code-soul.com
          </a>
        </div>
      </section>

      {/* 七、页脚 Footer - 五列 */}
      <footer className="border-t border-slate-200/80 bg-slate-50/80">
        <div className="max-w-7xl mx-auto px-5 sm:px-6 py-12 sm:py-16">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-8 lg:gap-6">
            {/* 第一列：品牌与愿景 */}
            <div className="col-span-2 md:col-span-1">
              <div className="flex items-center mb-3">
                <a href="https://github.com/CodeSoul-co" target="_blank" rel="noopener noreferrer" className="inline-flex items-center">
                  <img src="/codesoul-logo.png" alt="CodeSoul" className="h-10 sm:h-12 w-auto object-contain hover:opacity-80 transition-opacity" />
                </a>
              </div>
              <p className="text-xs text-slate-500 mb-2">Theta：洞见，先于思考</p>
              <a href="mailto:duanzhenke@code-soul.com" className="text-xs text-blue-600 hover:underline">
                duanzhenke@code-soul.com
              </a>
            </div>
            {/* 第二列：产品 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">产品</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="#changelog" className="hover:text-blue-600">功能更新日志</a></li>
                <li><a href="https://codesoul-co.github.io/THETA/zh/" target="_blank" rel="noopener noreferrer" className="hover:text-blue-600">API 文档</a></li>
                <li><Link href="/cases" className="hover:text-blue-600">案例库</Link></li>
              </ul>
            </div>
            {/* 第三列：支持与社区 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">支持与社区</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="#faq" className="hover:text-blue-600">帮助中心</a></li>
                <li><a href="#community" className="hover:text-blue-600">学术交流群</a></li>
                <li>
                  <button type="button" onClick={() => setShowCiteModal(true)} className="hover:text-blue-600 text-left">
                    如何引用 Theta？
                  </button>
                </li>
                <li><a href="#feedback" className="hover:text-blue-600">反馈建议</a></li>
              </ul>
            </div>
            {/* 第四列：社交媒体 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">社交媒体</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="https://github.com/CodeSoul-co/THETA" target="_blank" rel="noopener noreferrer" className="hover:text-blue-600">GitHub</a></li>
                <li><a href="#" className="hover:text-blue-600">X</a></li>
                <li><a href="#" className="hover:text-blue-600">Facebook</a></li>
              </ul>
            </div>
            {/* 第五列：关于 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">关于</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="#about" className="hover:text-blue-600">关于我们</a></li>
                <li><a href="#hiring" className="hover:text-blue-600">加入我们</a></li>
                <li><Link href="/security" className="hover:text-blue-600">安全白皮书</Link></li>
                <li><a href="#terms" className="hover:text-blue-600">用户协议</a></li>
                <li><a href="#privacy" className="hover:text-blue-600">隐私政策</a></li>
                <li className="text-slate-400 text-xs">ICP 备案号 / 公安联网备案</li>
              </ul>
            </div>
          </div>
          <p className="text-center text-slate-400 text-xs mt-10 pt-6 border-t border-slate-200/80">
            &copy; {new Date().getFullYear()} THETA · 零代码社科文本挖掘工作台
          </p>
        </div>
      </footer>

      {/* 如何引用 Theta - BibTeX 弹窗 */}
      <Dialog open={showCiteModal} onOpenChange={setShowCiteModal}>
        <DialogContent className="sm:max-w-lg bg-white">
          <DialogHeader>
            <DialogTitle>如何引用 Theta？</DialogTitle>
            <DialogDescription>在论文或报告中引用 THETA 时，可使用以下 BibTeX。</DialogDescription>
          </DialogHeader>
          <pre className="p-4 rounded-lg bg-slate-100 text-sm text-slate-800 overflow-x-auto font-mono whitespace-pre-wrap">
{`@software{theta2026,
  title = {THETA: 零代码社科文本挖掘工作台},
  author = {CodeSoul},
  year = {2026},
  url = {https://github.com/CodeSoul-co/THETA}
}`}
          </pre>
          <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(`@software{theta2026,\n  title = {THETA: 零代码社科文本挖掘工作台},\n  author = {CodeSoul},\n  year = {2026},\n  url = {https://github.com/CodeSoul-co/THETA}\n}`); }}>
            复制 BibTeX
          </Button>
        </DialogContent>
      </Dialog>

      {/* Login Modal */}
      <Dialog open={!OPEN_SOURCE_EDITION && showLoginModal} onOpenChange={(open) => { setShowLoginModal(open); if (!open) { setModalMode("login"); setRegSuccess(false); } }}>
        <DialogContent className="sm:max-w-md bg-white">
          <DialogHeader className="space-y-3 pb-2">
            <div className="flex items-center justify-center mb-2">
              <Image unoptimized
                src="/theta-logo.png"
                alt="THETA"
                width={1653}
                height={938}
                sizes="98px"
                className="h-14 w-auto object-contain"
              />
            </div>
            {/* 登录/注册 Tab 切换 */}
            {modalMode !== "reset" ? <div className="flex bg-slate-100 rounded-xl p-1 gap-1">
              <button
                type="button"
                onClick={handleGoToLogin}
                className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${
                  modalMode === "login"
                    ? "bg-white shadow-sm text-slate-800"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                登录
              </button>
              <button
                type="button"
                onClick={handleGoToRegister}
                className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${
                  modalMode === "register"
                    ? "bg-white shadow-sm text-slate-800"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                注册
              </button>
            </div> : <div className="flex items-center justify-between">
              <p className="text-base font-medium text-slate-800">重置密码</p>
              <button type="button" onClick={handleGoToLogin} className="text-sm text-blue-600 hover:underline">返回登录</button>
            </div>}
            <DialogTitle className="sr-only">
              {modalMode === "login" ? "登录 THETA" : modalMode === "register" ? "注册 THETA" : "重置密码"}
            </DialogTitle>
            <DialogDescription className="sr-only">
              {modalMode === "login" ? "登录您的账户" : modalMode === "register" ? "创建新账户" : "通过邮箱验证码重置密码"}
            </DialogDescription>
          </DialogHeader>

          {/* ========== 登录表单 ========== */}
          {modalMode === "login" && (
            <form onSubmit={handleLogin} className="space-y-5 py-4">
              {loginError && (
                <Alert variant="destructive" className="animate-in slide-in-from-top-2">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{loginError}</AlertDescription>
                </Alert>
              )}
              {loginNotice && (
                <Alert className="border-emerald-200 bg-emerald-50 text-emerald-700 animate-in slide-in-from-top-2">
                  <CheckCircle2 className="h-4 w-4" />
                  <AlertDescription>{loginNotice}</AlertDescription>
                </Alert>
              )}
              
              <div className="space-y-2">
                <Label htmlFor="login-phone" className="text-slate-700">
                  手机号
                </Label>
                <Input
                  id="login-phone"
                  type="tel"
                  inputMode="numeric"
                  placeholder="请输入注册时使用的手机号"
                  value={loginPhone}
                  onChange={(e) => setLoginPhone(e.target.value.replace(/\D/g, "").slice(0, 11))}
                  disabled={isLoggingIn}
                  className="h-11 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors"
                  autoComplete="tel"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="login-password" className="text-slate-700">
                  密码
                </Label>
                <div className="relative">
                  <Input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    placeholder="请输入密码"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    disabled={isLoggingIn}
                    className="h-11 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors pr-10"
                    autoComplete="current-password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                    tabIndex={-1}
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="remember-me"
                    checked={rememberMe}
                    onCheckedChange={(checked) => setRememberMe(checked === true)}
                    disabled={isLoggingIn}
                  />
                  <Label htmlFor="remember-me" className="text-sm font-normal cursor-pointer text-slate-600">
                    记住我
                  </Label>
                </div>
                <button type="button" onClick={() => toast.info("找回密码功能正在开发中", { duration: 5000 })} className="text-sm text-blue-600 hover:text-blue-700 hover:underline">
                  忘记密码？
                </button>
              </div>

              <Button 
                type="submit"
                className="w-full h-11 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition-all"
                disabled={isLoggingIn}
              >
                {isLoggingIn ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" />登录中...</>
                ) : "登录"}
              </Button>

              <div className="text-center text-sm text-slate-600">
                还没有账户？{" "}
                <button type="button" onClick={handleGoToRegister} className="text-blue-600 hover:text-blue-700 font-medium hover:underline">
                  立即注册
                </button>
              </div>
            </form>
          )}

          {/* ========== 注册表单 ========== */}
          {modalMode === "register" && (
            <div className="py-4">
              {regSuccess ? (
                <div className="flex flex-col items-center gap-3 py-8">
                  <CheckCircle2 className="w-14 h-14 text-green-500" />
                  <p className="text-base font-semibold text-slate-800">注册成功！</p>
                  <p className="text-sm text-slate-500">正在返回登录…</p>
                </div>
              ) : registerStep === 1 ? (
                <form onSubmit={(event) => { event.preventDefault(); void handleVerifyRegistrationCode() }} className="space-y-4" noValidate>
                  {regError && (
                    <Alert variant="destructive" className="animate-in slide-in-from-top-2">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{regError}</AlertDescription>
                    </Alert>
                  )}
                  {regNotice && (
                    <Alert className="border-emerald-200 bg-emerald-50 text-emerald-700 animate-in slide-in-from-top-2">
                      <CheckCircle2 className="h-4 w-4" />
                      <AlertDescription>{regNotice}</AlertDescription>
                    </Alert>
                  )}
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-phone" className="text-slate-700">手机号 <span className="text-red-500">*</span></Label>
                    <Input
                      id="reg-phone"
                      type="tel"
                      placeholder="请输入 11 位手机号"
                      value={regPhone}
                      onChange={(e) => { setRegPhone(e.target.value.replace(/\D/g, '').slice(0, 11)); setRegFieldErrors((p) => ({ ...p, phone: "" })) }}
                      disabled={isSendingCode || isVerifyingCode}
                      autoComplete="tel"
                      className={`h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors ${regFieldErrors.phone ? "border-red-400" : ""}`}
                    />
                    {regFieldErrors.phone && <p className="text-xs text-red-500">{regFieldErrors.phone}</p>}
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-code" className="text-slate-700">验证码 <span className="text-red-500">*</span></Label>
                    <div className="flex gap-3">
                      <Input
                        id="reg-code"
                        type="text"
                        inputMode="numeric"
                        placeholder="6 位数字"
                        value={regCode}
                        onChange={(e) => { setRegCode(e.target.value.replace(/\D/g, '').slice(0, 6)); setRegFieldErrors((p) => ({ ...p, code: "" })) }}
                        disabled={isVerifyingCode}
                        className={`h-10 bg-slate-50/50 border-slate-200 text-black focus:bg-white transition-colors ${regFieldErrors.code ? "border-red-400" : ""}`}
                        maxLength={6}
                      />
                      <Button
                        type="button"
                        onClick={handleSendCode}
                        disabled={isSendingCode || countdown > 0 || isVerifyingCode || !regPhone}
                        className="h-10 whitespace-nowrap px-4 flex-shrink-0"
                        variant="secondary"
                      >
                        {isSendingCode ? (
                          <><Loader2 className="mr-2 h-4 w-4 animate-spin" />发送中</>
                        ) : countdown > 0 ? (
                          `${countdown}秒后重发`
                        ) : (
                          "获取验证码"
                        )}
                      </Button>
                    </div>
                    {regFieldErrors.code && <p className="text-xs text-red-500">{regFieldErrors.code}</p>}
                  </div>
                  <Button type="submit" className="w-full h-11 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition-all" disabled={isVerifyingCode}>
                    {isVerifyingCode ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />验证中...</> : "下一步"}
                  </Button>
                  <div className="text-center text-sm text-slate-600">
                    已有账户？{" "}
                    <button type="button" onClick={handleGoToLogin} className="text-blue-600 hover:text-blue-700 font-medium hover:underline">返回登录</button>
                  </div>
                </form>
              ) : (
                <form onSubmit={handleRegister} className="space-y-4" noValidate>
                  {regError && (
                    <Alert variant="destructive" className="animate-in slide-in-from-top-2">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{regError}</AlertDescription>
                    </Alert>
                  )}
                  {regNotice && (
                    <Alert className="border-emerald-200 bg-emerald-50 text-emerald-700 animate-in slide-in-from-top-2">
                      <CheckCircle2 className="h-4 w-4" />
                      <AlertDescription>{regNotice}</AlertDescription>
                    </Alert>
                  )}
                  <p className="text-sm text-slate-500">手机号验证通过，请设置登录账号和密码。</p>
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-username" className="text-slate-700">用户名 <span className="text-red-500">*</span></Label>
                    <Input
                      id="reg-username"
                      type="text"
                      placeholder="2-64 位字母、数字、下划线、点或连字符"
                      value={regUsername}
                      onChange={(e) => { setRegUsername(e.target.value); setRegFieldErrors((p) => ({ ...p, username: "" })) }}
                      disabled={isRegistering}
                      autoComplete="username"
                      className={`h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors ${regFieldErrors.username ? "border-red-400" : ""}`}
                    />
                    {regFieldErrors.username && <p className="text-xs text-red-500">{regFieldErrors.username}</p>}
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-password" className="text-slate-700">密码 <span className="text-red-500">*</span></Label>
                    <div className="relative">
                      <Input id="reg-password" type={showRegPassword ? "text" : "password"} placeholder="6-20 位，字母+数字，可含 @" value={regPassword} onChange={(e) => { const next = normalizeThetaPasswordInput(e.target.value); setRegPassword(next); setRegFieldErrors((p) => ({ ...p, password: next && !isThetaPasswordValid(next) ? THETA_PASSWORD_POLICY_MESSAGE : "" })) }} disabled={isRegistering} autoComplete="new-password" className={`h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors pr-10 ${regFieldErrors.password ? "border-red-400" : ""}`} />
                      <button type="button" onClick={() => setShowRegPassword((v) => !v)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600" tabIndex={-1}>{showRegPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}</button>
                    </div>
                    {regFieldErrors.password && <p className="text-xs text-red-500">{regFieldErrors.password}</p>}
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="reg-confirm" className="text-slate-700">确认密码 <span className="text-red-500">*</span></Label>
                    <div className="relative">
                      <Input id="reg-confirm" type={showRegConfirm ? "text" : "password"} placeholder="再次输入密码" value={regConfirm} onChange={(e) => { setRegConfirm(normalizeThetaPasswordInput(e.target.value)); setRegFieldErrors((p) => ({ ...p, confirm: "" })) }} disabled={isRegistering} autoComplete="new-password" className={`h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors pr-10 ${regFieldErrors.confirm ? "border-red-400" : ""}`} />
                      <button type="button" onClick={() => setShowRegConfirm((v) => !v)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600" tabIndex={-1}>{showRegConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}</button>
                    </div>
                    {regFieldErrors.confirm && <p className="text-xs text-red-500">{regFieldErrors.confirm}</p>}
                  </div>
                  <div className="flex gap-3">
                    <Button type="button" variant="secondary" className="h-11 flex-1" disabled={isRegistering} onClick={() => setRegisterStep(1)}>返回修改手机号</Button>
                    <Button type="submit" className="h-11 flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition-all" disabled={isRegistering}>
                      {isRegistering ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />注册中...</> : "创建账户"}
                    </Button>
                  </div>
                  <div className="text-center text-sm text-slate-600">
                    已有账户？{" "}
                    <button type="button" onClick={handleGoToLogin} className="text-blue-600 hover:text-blue-700 font-medium hover:underline">返回登录</button>
                  </div>
                </form>
              )}
            </div>
          )}
          {modalMode === "reset" && (
            <form onSubmit={handleResetPassword} className="space-y-4 py-4">
              {resetError && <Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{resetError}</AlertDescription></Alert>}
              {resetNotice && <Alert className="border-emerald-200 bg-emerald-50 text-emerald-700"><CheckCircle2 className="h-4 w-4" /><AlertDescription>{resetNotice}</AlertDescription></Alert>}
              <div className="space-y-1.5">
                <Label htmlFor="reset-email">注册邮箱</Label>
                <Input id="reset-email" type="email" value={resetEmail} onChange={(e) => setResetEmail(e.target.value)} disabled={isResettingPassword} autoComplete="email" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="reset-code">验证码</Label>
                <div className="flex gap-3">
                  <Input id="reset-code" value={resetCode} onChange={(e) => setResetCode(e.target.value.replace(/\D/g, "").slice(0, 6))} maxLength={6} disabled={isResettingPassword} />
                  <Button type="button" variant="secondary" onClick={handleSendResetCode} disabled={isSendingResetCode || resetCountdown > 0 || isResettingPassword || !resetEmail} className="whitespace-nowrap">
                    {isSendingResetCode ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />发送中</> : resetCountdown > 0 ? `${resetCountdown}秒后重发` : "获取验证码"}
                  </Button>
                </div>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="reset-password">新密码</Label>
                <Input id="reset-password" type="password" value={resetNewPassword} onChange={(e) => setResetNewPassword(normalizeThetaPasswordInput(e.target.value))} placeholder="6-20 位，字母+数字，可含 @" disabled={isResettingPassword} autoComplete="new-password" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="reset-confirm">确认新密码</Label>
                <Input id="reset-confirm" type="password" value={resetConfirm} onChange={(e) => setResetConfirm(normalizeThetaPasswordInput(e.target.value))} disabled={isResettingPassword} autoComplete="new-password" />
              </div>
              <Button type="submit" className="w-full h-11" disabled={isResettingPassword}>
                {isResettingPassword ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />重置中...</> : "重置密码"}
              </Button>
            </form>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
