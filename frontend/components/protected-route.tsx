'use client';

import { OPEN_SOURCE_EDITION } from '@/lib/edition'

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/auth-context';
import { Loader2 } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, loading } = useAuth();
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    if (OPEN_SOURCE_EDITION) { router.replace("/workbench"); return; }
    if (!loading) {
      if (!isAuthenticated) {
        // Redirect to landing page (which has the login modal)
        router.replace('/?auth=login');
      } else {
        setIsChecking(false);
      }
    }
  }, [isAuthenticated, loading, router]);

  if (OPEN_SOURCE_EDITION) return null;

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
  if (!isAuthenticated) {
    return null;
  }

  return <>{children}</>;
}
