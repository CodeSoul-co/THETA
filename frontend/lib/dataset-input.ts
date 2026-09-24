export const isTextDocument = (name: string): boolean => /\.(txt|md|pdf|docx)$/iu.test(name)

export interface DatasetPreview {
  columns: string[]
  rows: string[][]
  inputKind?: 'table' | 'text'
  textColumn?: string
  totalRecords?: number
  segments?: { text: string; page?: number; paragraph?: number; chunk?: number; source_file?: string; table?: number; row?: number }[]
}
