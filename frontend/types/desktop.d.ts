import type { WebInferenceCatalog, WebInferenceSettingsUpdate } from '../components/theta-workbench/api/client'
export interface DesktopEmbedding {
  mode: 'local' | 'cloud'
  localModel: string
  localPath: string
  sbertPath: string
  provider: 'zhipu' | 'openai_compatible'
  baseUrl: string
  model: string
  dimensions: number | null
  apiKeyConfigured: boolean
}
export interface DesktopConfiguration {
  embedding: DesktopEmbedding
  home: string
  python: string
}
export interface DesktopUpdateState {
  status: 'disabled' | 'idle' | 'checking' | 'current' | 'available' | 'downloading' | 'ready' | 'installing' | 'error'
  currentVersion: string
  availableVersion?: string
  automatic: boolean
  manualInstall: boolean
  percent: number
  error?: string
}
declare global {
  interface Window {
    thetaDesktop?: {
      read(): Promise<DesktopConfiguration>
      catalog(): Promise<WebInferenceCatalog>
      saveInference(input: NonNullable<WebInferenceSettingsUpdate['llm']>): Promise<void>
      saveEmbedding(input: DesktopEmbedding & { apiKey?: string; clearApiKey?: boolean }): Promise<void>
      selectModel(): Promise<string | null>
      openModels(kind: 'qwen' | 'sbert'): Promise<void>
      openData(): Promise<void>
      updates: {
        state(): Promise<DesktopUpdateState>
        check(): Promise<DesktopUpdateState>
        download(): Promise<DesktopUpdateState>
        install(): Promise<void>
        configure(automatic: boolean): Promise<DesktopUpdateState>
        subscribe(callback: (state: DesktopUpdateState) => void): () => void
      }
      onOpenUpdates(callback: () => void): () => void
      onOpenSettings(callback: () => void): () => void
    }
  }
}
