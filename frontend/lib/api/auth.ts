/**
 * Authentication API Client
 * 适配 theta_1-main/api 后端认证接口
 *
 * 后端端点：
 *   POST /api/auth/login   → {access_token, token_type, expires_in, user: {username, role, created_at}}
 *   POST /api/auth/logout
 *   GET  /api/auth/me       → {username, role, expires_at}
 *   POST /api/auth/verify   → {valid, username, role, expires_at}
 */

import { apiFetch, API_BASE } from './config';
import { OPEN_SOURCE_EDITION } from '@/lib/edition';

// ==================== 类型定义 ====================

export interface User {
  id: string;
  username: string;
  email: string;
  phone?: string;
  full_name?: string;
  created_at: string;
  is_active: boolean;
  role?: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  expires_in: number;
  user?: User;
}

export interface LoginRequest {
  phone: string;
  password: string;
}

export interface ProfileUpdateRequest {
  email?: string;
  email_code?: string;
  full_name?: string;
}

export interface PasswordChangeRequest {
  current_password: string;
  new_password: string;
}

export interface RegisterRequest {
  username: string;
  phone: string;
  email?: string;
  password: string;
  full_name?: string;
  code: string;  // 短信验证码
}

export interface SendCodeRequest {
  email?: string;
  phone?: string;
  type: "register" | "reset_password" | "change_email";
}

export interface VerifyCodeRequest {
  email?: string;
  phone?: string;
  code: string;
  type?: "register" | "reset_password" | "change_email";
}

export interface ResetPasswordRequest {
  email: string;
  code: string;
  new_password: string;
}

function isLocalNoAuthMode(): boolean {
  if (OPEN_SOURCE_EDITION) return false;
  if (process.env.NEXT_PUBLIC_LOCAL_NO_AUTH === 'true') return true;
  if (process.env.NEXT_PUBLIC_LOCAL_NO_AUTH === 'false') return false;
  return false;
}

function localDevUser(): User {
  return {
    id: process.env.NEXT_PUBLIC_LOCAL_USER_ID || '00000000-0000-4000-8000-000000000001',
    username: process.env.NEXT_PUBLIC_LOCAL_USERNAME || 'theta_local_user',
    email: process.env.NEXT_PUBLIC_LOCAL_EMAIL || 'local@theta.dev',
    full_name: process.env.NEXT_PUBLIC_LOCAL_FULL_NAME || 'THETA 本地用户',
    created_at: new Date().toISOString(),
    is_active: true,
    role: process.env.NEXT_PUBLIC_LOCAL_ROLE || 'user',
  };
}

/** 将后端返回的用户/token 信息归一化为前端 User 格式 */
function normalizeUser(raw: any, fallbackUsername = 'unknown'): User {
  return {
    id: String(raw.id ?? ''),
    username: raw.username ?? fallbackUsername,
    email: raw.email ?? '',
    full_name: raw.full_name ?? raw.username ?? fallbackUsername,
    created_at: raw.created_at ?? raw.expires_at ?? '',
    is_active: raw.is_active ?? true,
    role: raw.role ?? 'user',
  };
}

// ==================== API ====================

export const AuthAPI = {
  isLocalNoAuthMode,

  /**
   * 注册新用户
   * POST /api/auth/register  body: {username, email, password, full_name?}
   */
  async register(data: RegisterRequest): Promise<User> {
    if (isLocalNoAuthMode()) {
      return localDevUser();
    }
    const raw = await apiFetch<any>(API_BASE, '/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
      timeoutMs: 12_000,
    });
    return normalizeUser(raw);
  },

  /**
   * 登录
   * THETA CLI Agent: POST /api/auth/login  body: {phone, password} (JSON)
   */
  async login(data: LoginRequest): Promise<Token> {
    if (isLocalNoAuthMode()) {
      const user = localDevUser();
      return {
        access_token: `theta-local.${user.username}.token`,
        token_type: 'bearer',
        expires_in: 86_400,
        user,
      };
    }
    const response = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      body: JSON.stringify(data),
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'same-origin',
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || '手机号或密码错误');
    }

    const raw = await response.json();

    const user: User = raw.user
      ? normalizeUser(raw.user, data.phone)
      : normalizeUser({ username: data.phone });

    return {
      access_token: raw.access_token,
      token_type: raw.token_type ?? 'bearer',
      expires_in: raw.expires_in ?? 86400,
      user,
    };
  },

  /** 获取当前用户信息。服务端会话是认证状态的唯一来源。 */
  async getCurrentUser(): Promise<User> {
    if (isLocalNoAuthMode()) {
      return localDevUser();
    }
    const raw = await apiFetch<any>(API_BASE, '/api/auth/me');
    const user = normalizeUser(raw);
    localStorage.setItem('user', JSON.stringify(user));
    return user;
  },

  /**
   * 验证 Token
   * POST /api/auth/verify  body: token string
   */
  async verifyToken(): Promise<{ valid: boolean; username: string; user_id: string }> {
    if (isLocalNoAuthMode()) {
      const user = localDevUser();
      return { valid: true, username: user.username, user_id: user.id };
    }
    try {
      const raw = await apiFetch<any>(API_BASE, '/api/auth/verify', {
        method: 'POST',
      });
      return {
        valid: raw.valid ?? true,
        username: raw.username ?? '',
        user_id: String(raw.user_id ?? ''),
      };
    } catch {
      try {
        const user = await this.getCurrentUser();
        return { valid: true, username: user.username, user_id: user.id };
      } catch {
        return { valid: false, username: '', user_id: '' };
      }
    }
  },

  async logout(): Promise<void> {
    if (isLocalNoAuthMode()) {
      return;
    }

    try {
      await apiFetch(API_BASE, '/api/auth/logout', { method: 'POST' });
    } catch {
      // 忽略后端登出失败
    }
    this.clearLocalAuthState();
  },

  isAuthenticated(): boolean {
    if (OPEN_SOURCE_EDITION) return false;
    if (isLocalNoAuthMode()) return true;
    return localStorage.getItem('theta_auth_state') === 'authenticated';
  },

  getToken(): string | null {
    if (isLocalNoAuthMode()) return `theta-local.${localDevUser().username}.token`;
    return null;
  },

  setAuth(_token: string, user: User): void {
    if (OPEN_SOURCE_EDITION) return;
    if (isLocalNoAuthMode()) {
      localStorage.setItem('user', JSON.stringify(localDevUser()));
      return;
    }
    localStorage.removeItem('access_token');
    localStorage.setItem('theta_auth_state', 'authenticated');
    localStorage.setItem('user', JSON.stringify(user));
  },

  getStoredUser(): User | null {
    if (OPEN_SOURCE_EDITION) return null;
    if (isLocalNoAuthMode()) {
      return localDevUser();
    }
    const s = localStorage.getItem('user');
    if (!s) return null;
    try { return JSON.parse(s); } catch { return null; }
  },

  clearLocalAuthState(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('theta_auth_state');
    localStorage.removeItem('user');
  },

  async updateProfile(data: ProfileUpdateRequest): Promise<User> {
    if (isLocalNoAuthMode()) {
      return localDevUser();
    }
    return apiFetch<User>(API_BASE, '/api/auth/profile', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async changePassword(data: PasswordChangeRequest): Promise<{ message: string }> {
    if (isLocalNoAuthMode()) {
      return { message: '本地免登录模式下不需要修改密码。' };
    }
    return apiFetch(API_BASE, '/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },



  async sendVerificationCode(data: SendCodeRequest): Promise<{ message: string; debug_code?: string }> {
    if (isLocalNoAuthMode()) {
      return { message: '本地免登录模式下已跳过验证码发送。' };
    }
    return apiFetch(API_BASE, '/api/auth/send-code', {
      method: 'POST',
      body: JSON.stringify(data),
      timeoutMs: 12_000,
    });
  },

  async verifyCode(data: VerifyCodeRequest): Promise<{ valid: boolean }> {
    if (isLocalNoAuthMode()) {
      return { valid: true };
    }
    return apiFetch(API_BASE, '/api/auth/verify-code', {
      method: 'POST',
      body: JSON.stringify({ type: 'register', ...data }),
      timeoutMs: 12_000,
    });
  },

  async resetPassword(data: ResetPasswordRequest): Promise<{ message: string }> {
    return apiFetch(API_BASE, '/api/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify(data),
      timeoutMs: 12_000,
    });
  },
};

export default AuthAPI;
