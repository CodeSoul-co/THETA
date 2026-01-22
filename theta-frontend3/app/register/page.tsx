'use client';

import { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/auth-context';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, AlertCircle, CheckCircle2, Eye, EyeOff, BrainCircuit, Check, X } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import Link from 'next/link';

// 密码强度检测
function getPasswordStrength(password: string): { score: number; label: string; color: string } {
  let score = 0;
  
  if (password.length >= 6) score += 1;
  if (password.length >= 8) score += 1;
  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score += 1;
  if (/\d/.test(password)) score += 1;
  if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) score += 1;
  
  if (score <= 1) return { score, label: '弱', color: 'bg-red-500' };
  if (score <= 2) return { score, label: '一般', color: 'bg-orange-500' };
  if (score <= 3) return { score, label: '中等', color: 'bg-yellow-500' };
  if (score <= 4) return { score, label: '强', color: 'bg-green-500' };
  return { score, label: '非常强', color: 'bg-emerald-500' };
}

export default function RegisterPage() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const { register, isAuthenticated, loading: authLoading } = useAuth();
  const router = useRouter();

  // 密码强度
  const passwordStrength = useMemo(() => getPasswordStrength(password), [password]);
  
  // 密码验证条件
  const passwordChecks = useMemo(() => ({
    minLength: password.length >= 6,
    hasLetter: /[a-zA-Z]/.test(password),
    hasNumber: /\d/.test(password),
    matches: password === confirmPassword && password.length > 0,
  }), [password, confirmPassword]);

  // 等待 auth 状态加载完成后再检查是否已登录
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      router.push('/');
    }
  }, [authLoading, isAuthenticated, router]);

  // 显示加载状态
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-slate-100">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-500">加载中...</p>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    // Validation
    if (username.length < 3) {
      setError('用户名长度至少为3个字符');
      return;
    }

    if (!/^[a-zA-Z0-9_\u4e00-\u9fa5]+$/.test(username)) {
      setError('用户名只能包含字母、数字、下划线和中文');
      return;
    }

    if (password !== confirmPassword) {
      setError('两次输入的密码不一致');
      return;
    }

    if (password.length < 6) {
      setError('密码长度至少为6个字符');
      return;
    }

    if (password.length > 128) {
      setError('密码长度不能超过128个字符');
      return;
    }

    setLoading(true);

    try {
      await register(username, email, password, fullName || undefined);
      setSuccess(true);
      setTimeout(() => {
        router.push('/');
      }, 1500);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : '注册失败，请稍后重试';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-slate-50 to-indigo-50 p-4">
      <Card className="w-full max-w-md shadow-xl border-slate-200/60">
        <CardHeader className="space-y-3 pb-4">
          <div className="flex items-center justify-center mb-2">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg">
              <BrainCircuit className="w-8 h-8 text-white" />
            </div>
          </div>
          <CardTitle className="text-2xl font-bold text-center bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            创建账户
          </CardTitle>
          <CardDescription className="text-center text-slate-500">
            注册以开始使用 THETA 智能分析平台
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert variant="destructive" className="animate-in slide-in-from-top-2">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {success && (
              <Alert className="bg-green-50 text-green-800 border-green-200 animate-in slide-in-from-top-2">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <AlertDescription>注册成功！正在跳转...</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="username" className="text-slate-700">用户名 <span className="text-red-500">*</span></Label>
              <Input
                id="username"
                type="text"
                placeholder="请输入用户名（3-50个字符）"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                minLength={3}
                maxLength={50}
                disabled={loading || success}
                className="h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors"
                autoComplete="username"
              />
              <p className="text-xs text-slate-400">支持字母、数字、下划线和中文</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="email" className="text-slate-700">邮箱 <span className="text-red-500">*</span></Label>
              <Input
                id="email"
                type="email"
                placeholder="请输入邮箱地址"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={loading || success}
                className="h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors"
                autoComplete="email"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="fullName" className="text-slate-700">姓名（可选）</Label>
              <Input
                id="fullName"
                type="text"
                placeholder="请输入您的姓名"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                disabled={loading || success}
                className="h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors"
                autoComplete="name"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-slate-700">密码 <span className="text-red-500">*</span></Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="请输入密码（6-128个字符）"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  maxLength={128}
                  disabled={loading || success}
                  className="h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors pr-10"
                  autoComplete="new-password"
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
              
              {/* 密码强度指示器 */}
              {password.length > 0 && (
                <div className="space-y-2 animate-in slide-in-from-top-1">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${passwordStrength.color} transition-all duration-300`}
                        style={{ width: `${(passwordStrength.score / 5) * 100}%` }}
                      />
                    </div>
                    <span className={`text-xs font-medium ${
                      passwordStrength.score <= 1 ? 'text-red-600' :
                      passwordStrength.score <= 2 ? 'text-orange-600' :
                      passwordStrength.score <= 3 ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      {passwordStrength.label}
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword" className="text-slate-700">确认密码 <span className="text-red-500">*</span></Label>
              <div className="relative">
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  placeholder="请再次输入密码"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  minLength={6}
                  maxLength={128}
                  disabled={loading || success}
                  className={`h-10 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors pr-10 ${
                    confirmPassword.length > 0 && password !== confirmPassword 
                      ? 'border-red-300 focus:border-red-500' 
                      : ''
                  }`}
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                  tabIndex={-1}
                >
                  {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* 密码验证清单 */}
            {password.length > 0 && (
              <div className="p-3 bg-slate-50 rounded-lg space-y-1.5 text-xs animate-in slide-in-from-top-2">
                <div className="flex items-center gap-2">
                  {passwordChecks.minLength ? (
                    <Check className="w-3.5 h-3.5 text-green-600" />
                  ) : (
                    <X className="w-3.5 h-3.5 text-slate-300" />
                  )}
                  <span className={passwordChecks.minLength ? 'text-green-700' : 'text-slate-500'}>
                    至少 6 个字符
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {passwordChecks.hasLetter ? (
                    <Check className="w-3.5 h-3.5 text-green-600" />
                  ) : (
                    <X className="w-3.5 h-3.5 text-slate-300" />
                  )}
                  <span className={passwordChecks.hasLetter ? 'text-green-700' : 'text-slate-500'}>
                    包含字母
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {passwordChecks.hasNumber ? (
                    <Check className="w-3.5 h-3.5 text-green-600" />
                  ) : (
                    <X className="w-3.5 h-3.5 text-slate-300" />
                  )}
                  <span className={passwordChecks.hasNumber ? 'text-green-700' : 'text-slate-500'}>
                    包含数字
                  </span>
                </div>
                {confirmPassword.length > 0 && (
                  <div className="flex items-center gap-2">
                    {passwordChecks.matches ? (
                      <Check className="w-3.5 h-3.5 text-green-600" />
                    ) : (
                      <X className="w-3.5 h-3.5 text-red-500" />
                    )}
                    <span className={passwordChecks.matches ? 'text-green-700' : 'text-red-600'}>
                      {passwordChecks.matches ? '密码匹配' : '密码不匹配'}
                    </span>
                  </div>
                )}
              </div>
            )}

            <Button 
              type="submit" 
              className="w-full h-11 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition-all mt-2"
              disabled={loading || success}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  注册中...
                </>
              ) : success ? (
                <>
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                  注册成功
                </>
              ) : (
                '注册'
              )}
            </Button>

            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-200"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-white px-2 text-slate-400">或</span>
              </div>
            </div>

            <div className="text-center text-sm text-slate-600">
              已有账户？{' '}
              <Link href="/login" className="text-blue-600 hover:text-blue-700 font-medium hover:underline">
                立即登录
              </Link>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
