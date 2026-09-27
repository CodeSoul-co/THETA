export type DatasetRole = 'all' | 'train' | 'validation' | 'test'
export const DATASET_LABELS: Record<DatasetRole, string> = { all: '全量数据', train: '训练集', validation: '验证集', test: '测试集' }
export type SplitGroup = { count: number; metrics: Record<string, unknown>; topicProportions: number[]; maxTopicStd?: number }
export type SplitReport = { model: string; testUsesAllData: boolean; method: string; mode: string; seed: number; groups: Record<DatasetRole, SplitGroup> }
export function filesForDataset<T extends { name: string }>(files: T[], role: DatasetRole): T[] {
  return files.filter(file => {
    const match = /(?:^|\/)splits\/(all|train|validation|test)\//.exec(file.name)
    return role === 'all' ? !match : match?.[1] === role
  })
}
export function formatSplitValue(value: number): string {
  return Number.isFinite(value) ? Number(value.toPrecision(8)).toString() : '不可用'
}
