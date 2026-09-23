export type SetupTab = 'inference' | 'embedding'

export const LARGE_DATASET_BYTES = 20_000_000
const computeIntensiveModels = new Set(['theta', 'ctm', 'bertopic', 'dtm', 'nvdm', 'gsm', 'prodlda', 'etm'])

export function computationNotice(sizeBytes = 0, models: string[] = [], locale = 'zh-CN'): string | undefined {
  const large = sizeBytes > LARGE_DATASET_BYTES
  const heavy = models.some(model => computeIntensiveModels.has(model.toLowerCase()))
  if (!large && !heavy) return
  const zh = locale === 'zh-CN'
  const reason = large && heavy
    ? (zh ? '数据超过 20 MB，且所选模型计算量较大。' : 'The data exceeds 20 MB and the selected models require substantial computation.')
    : large ? (zh ? '数据超过 20 MB。' : 'The data exceeds 20 MB.')
      : (zh ? '所选模型计算量较大。' : 'The selected models require substantial computation.')
  return reason + (zh
    ? '预处理、嵌入与训练可能耗时较久，尤其在 CPU 上；实际时间取决于硬件、数据规模和参数。请保持应用与电脑运行，查看任务进度，避免重复提交。'
    : 'Preprocessing, embeddings, and training may take a while, especially on CPU. Duration depends on hardware, data size, and parameters. Keep the app and computer running, check task progress, and avoid submitting twice.')
}

/** Classify known setup failures only; retain unrelated errors for diagnosis. */
export function errorGuidance(raw: string, locale = 'zh-CN'): { message: string; settingsTab?: SetupTab } {
  const zh = locale === 'zh-CN'
  if (/modelAssetsReady["']?\s*:\s*false|(?:模型|model|embedding|嵌入).{0,45}(?:未下载|未找到|不存在|尚未准备好|not found|weights.*missing)|(?:QWEN_MODEL|SBERT_MODEL_PATH).{0,40}(?:未配置|missing|not set)/iu.test(raw)) {
    return { settingsTab: 'embedding', message: zh
      ? '本地嵌入模型尚未准备好。请到「设置 → 嵌入模型」下载兼容模型，并选择包含配置、分词器和权重的完整目录。安装包不包含模型权重；THETA 零样本分析也可改用已配置的云端嵌入。'
      : 'The local embedding model is not ready. In Settings → Embeddings, download a compatible model and select its complete configuration, tokenizer, and weights directory. Model weights are not bundled. THETA zero-shot can also use configured cloud embeddings.' }
  }
  if (/(?:api.?key|密钥|供应商).{0,45}(?:missing|not configured|invalid|未配置|未填写|无效)|(?:missing|invalid).{0,20}api.?key|(?:未配置|尚未配置).{0,35}API_KEY|需要完整的自定义 API|云端.{0,25}(?:未配置|未就绪)/iu.test(raw)) {
    const embedding = /embedding|嵌入|云端/iu.test(raw)
    return { settingsTab: embedding ? 'embedding' : 'inference', message: zh
      ? `${embedding ? '云端嵌入' : '对话模型'} API 未配置完整或密钥无效。请到「设置 → ${embedding ? '嵌入模型' : '模型 API'}」填写服务地址、模型名称和自己的 API Key，保存后重试。`
      : `The ${embedding ? 'embedding' : 'conversation-model'} API configuration is incomplete or its key is invalid. Open Settings → ${embedding ? 'Embeddings' : 'Model API'}, enter the endpoint, model name, and your API key, then save and retry.` }
  }
  if (/out of memory|内存不足|cannot allocate memory/iu.test(raw)) return { message: zh
    ? '可用内存不足。请关闭其他占用内存的应用，减小批大小或数据规模，或选择更小的模型后重试。'
    : 'Not enough memory. Close memory-intensive apps, reduce the batch or dataset size, or choose a smaller model before retrying.' }
  return { message: raw }
}

export function openSetup(tab: SetupTab): void {
  if (tab === 'embedding' && !window.thetaDesktop) {
    window.open('https://github.com/CodeSoul-co/THETA/blob/main/agent/docs/local-embedding.md', '_blank', 'noopener,noreferrer')
    return
  }
  window.dispatchEvent(new CustomEvent('theta:open-settings', { detail: tab }))
}
