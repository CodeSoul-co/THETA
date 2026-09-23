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
      onOpenSettings(callback: () => void): () => void
    }
  }
}
