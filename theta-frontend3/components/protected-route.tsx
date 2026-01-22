'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/auth-context';
import { AuthAPI } from '@/lib/api/auth';
import { Loader2 } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, loading } = useAuth();
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    // 如果认证检查完成，检查是否有 token（应对状态更新延迟的情况）
    if (!loading) {
      const hasToken = AuthAPI.isAuthenticated();
      const actuallyAuthenticated = isAuthenticated || hasToken;
      
      if (!actuallyAuthenticated) {
        router.replace('/login');
      } else {
        setIsChecking(false);
      }
    }
  }, [isAuthenticated, loading, router]);

  // 显示加载状态
  if (loading || isChecking) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-500">加载中...</p>
        </div>
      </div>
    );
  }

  // 最终检查：如果仍然未认证，返回 null（会在 useEffect 中重定向）
  const hasToken = AuthAPI.isAuthenticated();
  if (!isAuthenticated && !hasToken) {
    return null;
  }

  return <>{children}</>;
}
