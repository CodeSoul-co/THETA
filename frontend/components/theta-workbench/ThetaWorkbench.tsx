'use client'

import { AppRoot } from './App.tsx'
import { InferenceSettingsProvider } from './inference-settings.tsx'
import { PreferencesProvider } from './preferences.tsx'
import { useAuth } from '@/contexts/auth-context'
import css from './ThetaWorkbench.module.css'

export function ThetaWorkbench({ initialMode }: { initialMode?: 'conversation' | 'manual' } = {}): React.ReactElement {
  const { user } = useAuth()
  const accountKey = user?.id ?? 'anonymous'
  return (
    <div className={css.host}>
      <PreferencesProvider>
        <InferenceSettingsProvider>
          <AppRoot key={String(accountKey)} initialMode={initialMode} />
        </InferenceSettingsProvider>
      </PreferencesProvider>
    </div>
  )
}
