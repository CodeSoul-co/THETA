export const splitRoles = ['train', 'validation', 'test'] as const
export type SplitRole = typeof splitRoles[number]
export type SplitSource = { fileId: string; textColumn: string; timeColumn?: string; labelColumn?: string; covariates?: string[] }
export type DataSplit = { enabled: boolean; mode: 'ratio' | 'upload'; method: 'random' | 'sequential'; ratios: number[]; seed: number; sources: Partial<Record<SplitRole, SplitSource>> }
export const defaultDataSplit: DataSplit = { enabled: false, mode: 'ratio', method: 'random', ratios: [0.7, 0.2, 0.1], seed: 42, sources: {} }
export function splitError(value: DataSplit): string | undefined {
  if (!value.enabled) return
  if (value.mode === 'upload') {
    if (splitRoles.some(role => !value.sources[role]?.fileId || !value.sources[role]?.textColumn)) return '请分别上传并选择训练、验证和测试集的正文列。'
    if (new Set(splitRoles.map(role => value.sources[role]!.fileId)).size !== 3) return '训练、验证和测试集应选择三份不同的数据。'
  } else if (value.ratios.length !== 3 || value.ratios.some(v => !Number.isFinite(v) || v <= 0) || Math.abs(value.ratios.reduce((a, b) => a + b, 0) - 1) > 1e-8) return '三份数据的比例都须大于 0，且合计为 100%。'
  if (!Number.isInteger(value.seed) || value.seed < 0 || value.seed >= 2**32) return '随机种子必须为 0–4294967295 的整数。'
}
