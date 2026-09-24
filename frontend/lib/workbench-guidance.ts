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
  if (/BOW vocabulary is empty|没有可用词语|empty vocabulary/iu.test(raw)) return { message: zh
    ? '失败原因：正文分词后没有可用词语。可能是正文列选错、内容太少、只有数字或停用词过滤过多。请检查正文预览和停用词，增加有效文本后重新分析。'
    : 'No usable words remain after tokenization. Check the text column, document contents and stopwords, and add meaningful text before retrying.' }
  if (/File is not a zip file|Package not found at|文件不完整|文件压缩内容损坏/iu.test(raw)) return { message: zh
    ? '失败原因：Office 文件内容不完整或格式无法识别。可能是旧版上传截断、原文件损坏、加密，或只修改了扩展名。请确认原文件能正常打开，另存为标准 XLSX / DOCX，再使用“重新上传 / 更换数据”。'
    : 'The Office file is incomplete or unrecognized. It may be truncated, damaged, encrypted, or incorrectly renamed. Open and save the original as XLSX or DOCX, then upload it again.' }
  if (/PermissionError|EACCES|EPERM|Permission denied|访问权限/iu.test(raw)) return { message: zh
    ? '失败原因：无法读写文件。请关闭占用文件的程序，检查磁盘剩余空间及数据目录权限；可在应用设置中查看实际数据保存位置。'
    : 'The file cannot be read or written. Close apps holding the file and check disk space and data-folder permissions. Settings shows the active data location.' }
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
