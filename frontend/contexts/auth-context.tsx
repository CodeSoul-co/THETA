'use client';

import { OPEN_SOURCE_EDITION } from '@/lib/edition'

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { AuthAPI, User, ProfileUpdateRequest, PasswordChangeRequest, SendCodeRequest, VerifyCodeRequest, ResetPasswordRequest } from '@/lib/api/auth';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (phone: string, password: string, rememberMe?: boolean) => Promise<void>;
  register: (username: string, phone: string, password: string, fullName?: string, code?: string) => Promise<void>;
  logout: () => Promise<void>;
  updateProfile: (data: ProfileUpdateRequest) => Promise<void>;
  changePassword: (data: PasswordChangeRequest) => Promise<void>;
  sendVerificationCode: (data: SendCodeRequest) => Promise<{ message: string; debug_code?: string }>;
  verifyCode: (data: VerifyCodeRequest) => Promise<boolean>;
  resetPassword: (data: ResetPasswordRequest) => Promise<void>;
  refreshUser: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const refreshUser = useCallback(async () => {
    if (OPEN_SOURCE_EDITION) return;
    try {
      const userData = await AuthAPI.getCurrentUser();
      setUser(userData);
      AuthAPI.setAuth('http-only-cookie', userData);
    } catch {
      // The server session is authoritative. A stale browser cache must never
      // keep an account visible after the session can no longer be verified.
      AuthAPI.clearLocalAuthState();
      setUser(null);
    }
  }, []);

  useEffect(() => {
    if (OPEN_SOURCE_EDITION) { setLoading(false); return; }
    const checkAuth = async () => {
      try {
        const userData = await AuthAPI.getCurrentUser();
        setUser(userData);
        AuthAPI.setAuth('http-only-cookie', userData);
      } catch {
        AuthAPI.clearLocalAuthState();
        setUser(null);
      }
      setLoading(false);
    };

    checkAuth();
  }, [refreshUser]);

  const login = async (phone: string, password: string, rememberMe: boolean = false) => {
    try {
      const tokenResponse = await AuthAPI.login({ phone, password });
      
      if (!tokenResponse.access_token) {
        throw new Error('登录响应中没有 token');
      }
      
      // 使用登录响应中的用户信息（如果存在），否则获取
      let userData: User;
      if (tokenResponse.user) {
        userData = tokenResponse.user;
      } else {
        userData = await AuthAPI.getCurrentUser();
      }
      
      // 如果选择了"记住我"，延长 token 有效期（实际应该在后端实现，这里只是前端标记）
      if (rememberMe) {
        localStorage.setItem('remember_me', 'true');
      } else {
        localStorage.removeItem('remember_me');
      }
      
      // 保存 token 和用户信息
      AuthAPI.setAuth(tokenResponse.access_token, userData);
      setUser(userData);
      
    } catch (error) {
      // 手机号/密码错误是预期行为，不打印到控制台；仅记录意外错误
      const msg = error instanceof Error ? error.message : String(error);
      const isCredentialError = /用户名|密码|401|Incorrect|Unauthorized/i.test(msg);
      if (!isCredentialError) {
        console.error('[Auth] 登录失败:', error);
      }
      AuthAPI.logout();
      setUser(null);
      throw error;
    }
  };

  const register = async (username: string, phone: string, password: string, fullName?: string, code?: string) => {
    await AuthAPI.register({ username, phone, password, full_name: fullName, code: code || '' });
  };

  const logout = useCallback(async () => {
    await AuthAPI.logout();
    localStorage.removeItem('remember_me');
    localStorage.removeItem('remembered_username');
    localStorage.removeItem('remembered_phone');
    setUser(null);
    router.push('/?auth=login');
  }, [router]);

  const updateProfile = async (data: ProfileUpdateRequest) => {
    const updatedUser = await AuthAPI.updateProfile(data);
    setUser(updatedUser);
    // Update stored user
    AuthAPI.setAuth('http-only-cookie', updatedUser);
  };

  const changePassword = async (data: PasswordChangeRequest) => {
    await AuthAPI.changePassword(data);
  };

  const sendVerificationCode = async (data: SendCodeRequest) => {
    return AuthAPI.sendVerificationCode(data);
  };

  const verifyCode = async (data: VerifyCodeRequest): Promise<boolean> => {
    const result = await AuthAPI.verifyCode(data);
    return result.valid;
  };

  const resetPassword = async (data: ResetPasswordRequest): Promise<void> => {
    await AuthAPI.resetPassword(data);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        logout,
        updateProfile,
        changePassword,
        sendVerificationCode,
        verifyCode,
        resetPassword,
        refreshUser,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
