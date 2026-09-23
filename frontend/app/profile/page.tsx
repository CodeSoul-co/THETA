"use client"

import { useState, useEffect } from "react"
import { ProtectedRoute } from "@/components/protected-route"
import { useAuth } from "@/contexts/auth-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { AlertCircle, CheckCircle2, User as UserIcon, Lock, Calendar, Shield, Mail } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { isThetaPasswordValid, normalizeThetaPasswordInput, THETA_PASSWORD_POLICY_MESSAGE } from "@/lib/auth/password-policy"

export default function ProfilePage() {
  return (
    <ProtectedRoute>
      <ProfileContent />
    </ProtectedRoute>
  )
}

function ProfileContent() {
  const { user, updateProfile, changePassword, sendVerificationCode, logout } = useAuth()

  // 统计数据
  const [stats, setStats] = useState({
    projects: 0,
    datasets: 0,
    tasks: 0,
  })
  const [loadingStats, setLoadingStats] = useState(true)

  // 编辑表单状态
  const [fullName, setFullName] = useState("")
  const [email, setEmail] = useState("")
  const [isEditing, setIsEditing] = useState(false)
  const [updateStatus, setUpdateStatus] = useState<{ type: "success" | "error"; message: string } | null>(null)
  const [isUpdating, setIsUpdating] = useState(false)
  const [emailCode, setEmailCode] = useState("")
  const [isSendingEmailCode, setIsSendingEmailCode] = useState(false)

  // 修改密码状态
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [passwordStatus, setPasswordStatus] = useState<{ type: "success" | "error"; message: string } | null>(null)
  const [isChangingPassword, setIsChangingPassword] = useState(false)

  // 初始化表单
  useEffect(() => {
    if (user) {
      setFullName(user.full_name || "")
      setEmail(user.email || "")
    }
  }, [user])

  // 加载统计数据
  useEffect(() => {
    const loadStats = async () => {
      try {
        const [projects, datasets, tasks] = await Promise.all([
          ETMAgentAPI.getProjects().catch(() => []),
          ETMAgentAPI.getDatasets().catch(() => []),
          ETMAgentAPI.getTasks({ limit: 100 }).catch(() => []),
        ])
        setStats({
          projects: projects.length,
          datasets: datasets.length,
          tasks: tasks.length,
        })
      } catch (error) {
        console.error("Failed to load stats:", error)
      } finally {
        setLoadingStats(false)
      }
    }
    loadStats()
  }, [])

  // 获取用户首字母
  const getInitials = () => {
    if (user?.full_name) {
      return user.full_name.charAt(0).toUpperCase()
    }
    if (user?.username) {
      return user.username.charAt(0).toUpperCase()
    }
    return "U"
  }

  // 格式化创建时间
  const formatCreatedAt = () => {
    if (!user?.created_at) return "未知"
    try {
      const date = new Date(user.created_at)
      return date.toLocaleDateString("zh-CN", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    } catch {
      return user.created_at
    }
  }

  // 获取角色标签
  const getRoleLabel = (role?: string) => {
    const roleMap: Record<string, string> = {
      admin: "管理员",
      user: "普通用户",
    }
    return roleMap[role || "user"] || role || "普通用户"
  }

  // 处理更新个人资料
  const handleUpdateProfile = async () => {
    setUpdateStatus(null)
    const normalizedEmail = email.trim().toLowerCase()
    const emailChanged = normalizedEmail !== (user?.email || "").trim().toLowerCase()
    if (emailChanged && !/^\d{6}$/.test(emailCode)) {
      setUpdateStatus({ type: "error", message: "修改邮箱需要输入发送到新邮箱的 6 位验证码" })
      return
    }
    setIsUpdating(true)

    try {
      await updateProfile({
        full_name: fullName.trim() || undefined,
        email: normalizedEmail || undefined,
        email_code: emailChanged ? emailCode : undefined,
      })
      setUpdateStatus({
        type: "success",
        message: "个人资料更新成功",
      })
      setIsEditing(false)
      setEmailCode("")
    } catch (error: any) {
      setUpdateStatus({
        type: "error",
        message: error.message || "更新失败，请重试",
      })
    } finally {
      setIsUpdating(false)
    }
  }

  const handleSendEmailCode = async () => {
    const normalizedEmail = email.trim().toLowerCase()
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
      setUpdateStatus({ type: "error", message: "请先输入有效的新邮箱地址" })
      return
    }
    setUpdateStatus(null)
    setIsSendingEmailCode(true)
    try {
      await sendVerificationCode({ email: normalizedEmail, type: "change_email" })
      setUpdateStatus({ type: "success", message: "验证码已发送到新邮箱，有效期内请完成保存" })
    } catch (error: unknown) {
      setUpdateStatus({ type: "error", message: error instanceof Error ? error.message : "验证码发送失败" })
    } finally {
      setIsSendingEmailCode(false)
    }
  }

  // 处理修改密码
  const handleChangePassword = async () => {
    setPasswordStatus(null)

    if (newPassword !== confirmPassword) {
      setPasswordStatus({
        type: "error",
        message: "两次输入的密码不一致",
      })
      return
    }

    if (!isThetaPasswordValid(newPassword)) {
      setPasswordStatus({
        type: "error",
        message: THETA_PASSWORD_POLICY_MESSAGE,
      })
      return
    }

    setIsChangingPassword(true)

    try {
      await changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      })
      setPasswordStatus({
        type: "success",
        message: "密码修改成功，请重新登录",
      })
      // 清空表单
      setCurrentPassword("")
      setNewPassword("")
      setConfirmPassword("")
      // 后端已撤销全部会话，前端同步清除 HttpOnly Cookie 和本地登录标记。
      setTimeout(() => {
        void logout()
      }, 3000)
    } catch (error: any) {
      setPasswordStatus({
        type: "error",
        message: error.message || "密码修改失败，请检查当前密码是否正确",
      })
    } finally {
      setIsChangingPassword(false)
    }
  }

  return (
    <div className="container max-w-4xl py-8 mx-auto px-4">
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">个人资料</h1>
        <p className="text-slate-500">管理您的账户信息和设置</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* 左侧：用户概览 */}
        <div className="md:col-span-1">
          <Card>
            <CardHeader className="text-center">
              <div className="flex justify-center mb-4">
                <Avatar className="h-24 w-24 ring-4 ring-white shadow-lg">
                  <AvatarFallback className="bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 text-white text-3xl font-semibold">
                    {getInitials()}
                  </AvatarFallback>
                </Avatar>
              </div>
              <CardTitle className="text-xl">{user?.full_name || user?.username}</CardTitle>
              <CardDescription>{user?.email}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-500">角色</span>
                  <Badge variant="secondary" className={user?.role === "admin" ? "bg-blue-100 text-blue-700" : ""}>
                    {getRoleLabel(user?.role)}
                  </Badge>
                </div>
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <Calendar className="h-4 w-4 text-slate-400" />
                  <span>注册于 {formatCreatedAt()}</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <Shield className="h-4 w-4 text-slate-400" />
                  <span>{user?.is_active ? "账户已激活" : "账户未激活"}</span>
                </div>
              </div>

              <Separator className="my-6" />

              {/* 使用统计 */}
              <div>
                <h3 className="text-sm font-medium text-slate-900 mb-3">使用统计</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-500">已保存项目</span>
                    <span className="font-semibold text-slate-900">
                      {loadingStats ? "-" : stats.projects}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-500">上传数据集</span>
                    <span className="font-semibold text-slate-900">
                      {loadingStats ? "-" : stats.datasets}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-500">训练任务</span>
                    <span className="font-semibold text-slate-900">
                      {loadingStats ? "-" : stats.tasks}
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 右侧：编辑和修改密码 */}
        <div className="md:col-span-2">
          <Tabs defaultValue="profile" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="profile" className="gap-2">
                <UserIcon className="h-4 w-4" />
                个人信息
              </TabsTrigger>
              <TabsTrigger value="security" className="gap-2">
                <Lock className="h-4 w-4" />
                安全设置
              </TabsTrigger>
            </TabsList>

            {/* 个人信息编辑 */}
            <TabsContent value="profile">
              <Card>
                <CardHeader>
                  <CardTitle>个人信息</CardTitle>
                  <CardDescription>修改您的个人资料信息</CardDescription>
                </CardHeader>
                <CardContent>
                  {updateStatus && (
                    <Alert className={updateStatus.type === "success" ? "bg-green-50 border-green-200 mb-6" : "bg-red-50 border-red-200 mb-6"}>
                      {updateStatus.type === "success" ? (
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      ) : (
                        <AlertCircle className="h-4 w-4 text-red-600" />
                      )}
                      <AlertDescription className={updateStatus.type === "success" ? "text-green-700" : "text-red-700"}>
                        {updateStatus.message}
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="space-y-4">
                    <div className="grid grid-cols-4 gap-4 items-center">
                      <Label htmlFor="username" className="text-right col-span-4 sm:col-span-1">
                        用户名
                      </Label>
                      <div className="col-span-4 sm:col-span-3">
                        <Input
                          id="username"
                          value={user?.username || ""}
                          disabled
                          className="bg-slate-50 text-slate-500"
                        />
                        <p className="text-xs text-slate-500 mt-1">用户名不可修改</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 items-center">
                      <Label htmlFor="fullName" className="text-right col-span-4 sm:col-span-1">
                        姓名
                      </Label>
                      <div className="col-span-4 sm:col-span-3">
                        <Input
                          id="fullName"
                          value={fullName}
                          onChange={(e) => setFullName(e.target.value)}
                          disabled={!isEditing}
                          placeholder="请输入您的姓名"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 items-center">
                      <Label htmlFor="email" className="text-right col-span-4 sm:col-span-1">
                        邮箱
                      </Label>
                      <div className="col-span-4 sm:col-span-3">
                        <Input
                          id="email"
                          type="email"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          disabled={!isEditing}
                          placeholder="your@email.com"
                        />
                        {isEditing && email.trim().toLowerCase() !== (user?.email || "").trim().toLowerCase() && (
                          <div className="mt-2 flex gap-2">
                            <Input
                              value={emailCode}
                              onChange={(event) => setEmailCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
                              placeholder="新邮箱验证码"
                              maxLength={6}
                              inputMode="numeric"
                            />
                            <Button type="button" variant="outline" onClick={handleSendEmailCode} disabled={isSendingEmailCode} className="whitespace-nowrap">
                              {isSendingEmailCode ? "发送中..." : "发送验证码"}
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 items-center">
                      <Label className="text-right col-span-4 sm:col-span-1">
                        用户ID
                      </Label>
                      <div className="col-span-4 sm:col-span-3">
                        <Input
                          value={user?.id?.toString() || ""}
                          disabled
                          className="bg-slate-50 text-slate-500"
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-end gap-2">
                  {isEditing ? (
                    <>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setIsEditing(false)
                          // 重置为原始值
                          if (user) {
                            setFullName(user.full_name || "")
                            setEmail(user.email || "")
                          }
                          setEmailCode("")
                          setUpdateStatus(null)
                        }}
                      >
                        取消
                      </Button>
                      <Button
                        onClick={handleUpdateProfile}
                        disabled={isUpdating}
                      >
                        {isUpdating ? "保存中..." : "保存修改"}
                      </Button>
                    </>
                  ) : (
                    <Button onClick={() => setIsEditing(true)}>
                      编辑资料
                    </Button>
                  )}
                </CardFooter>
              </Card>
            </TabsContent>

            {/* 安全设置 - 修改密码 */}
            <TabsContent value="security">
              <Card>
                <CardHeader>
                  <CardTitle>修改密码</CardTitle>
                  <CardDescription>更新您的账户密码以保证账户安全</CardDescription>
                </CardHeader>
                <CardContent>
                  {passwordStatus && (
                    <Alert className={passwordStatus.type === "success" ? "bg-green-50 border-green-200 mb-6" : "bg-red-50 border-red-200 mb-6"}>
                      {passwordStatus.type === "success" ? (
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      ) : (
                        <AlertCircle className="h-4 w-4 text-red-600" />
                      )}
                      <AlertDescription className={passwordStatus.type === "success" ? "text-green-700" : "text-red-700"}>
                        {passwordStatus.message}
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="space-y-4">
                    <div className="grid grid-cols-4 gap-4 items-center">
                      <Label htmlFor="currentPassword" className="text-right col-span-4 sm:col-span-1">
                        当前密码
                      </Label>
                      <div className="col-span-4 sm:col-span-3">
                        <Input
                          id="currentPassword"
                          type="password"
                          value={currentPassword}
                          onChange={(e) => setCurrentPassword(e.target.value)}
                          placeholder="请输入当前密码"
                        />
                      </div>
                    </div>

                    <Separator className="my-2" />

                    <div className="grid grid-cols-4 gap-4 items-center">
                      <Label htmlFor="newPassword" className="text-right col-span-4 sm:col-span-1">
                        新密码
                      </Label>
                      <div className="col-span-4 sm:col-span-3">
                        <Input
                          id="newPassword"
                          type="password"
                          value={newPassword}
                          onChange={(e) => setNewPassword(normalizeThetaPasswordInput(e.target.value))}
                          placeholder="6-20 位，字母+数字，可含 @"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 items-center">
                      <Label htmlFor="confirmPassword" className="text-right col-span-4 sm:col-span-1">
                        确认密码
                      </Label>
                      <div className="col-span-4 sm:col-span-3">
                        <Input
                          id="confirmPassword"
                          type="password"
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(normalizeThetaPasswordInput(e.target.value))}
                          placeholder="再次输入新密码"
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-end">
                  <Button
                    onClick={handleChangePassword}
                    disabled={isChangingPassword}
                    className="gap-2"
                  >
                    <Lock className="h-4 w-4" />
                    {isChangingPassword ? "修改中..." : "修改密码"}
                  </Button>
                </CardFooter>
              </Card>

              <Card className="mt-6">
                <CardHeader>
                  <CardTitle>账户安全提示</CardTitle>
                  <CardDescription>帮助您保护账户安全的一些建议</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm text-slate-600 list-disc pl-5">
                    <li>使用 6-20 位密码，包含英文字母和数字，可使用 @</li>
                    <li>不要在多个网站使用相同的密码</li>
                    <li>定期更换密码可以提高账户安全性</li>
                    <li>不要将密码分享给他人</li>
                  </ul>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
