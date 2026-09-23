import {
  Database,
  FileArchive,
  FileJson2,
  FileSpreadsheet,
  FileText,
  FileType2,
} from 'lucide-react'
import css from '../styles/app.module.css'

const extensionOf = (filename: string): string => filename.split('.').at(-1)?.toLowerCase() ?? ''

export const DatasetFileIcon = ({ filename, size = 18 }: { filename: string; size?: number }): React.ReactElement => {
  const extension = extensionOf(filename)
  const Icon = extension === 'csv' || extension === 'tsv' || extension === 'xlsx' || extension === 'xls'
    ? FileSpreadsheet
    : extension === 'json' || extension === 'jsonl' || extension === 'ndjson'
      ? FileJson2
      : extension === 'pdf'
        ? FileText
        : extension === 'doc' || extension === 'docx'
          ? FileType2
          : extension === 'parquet'
            ? Database
            : extension === 'zip' || extension === 'tar' || extension === 'gz'
              ? FileArchive
              : FileText

  return <span className={css.datasetFileIcon} data-file-type={extension || 'file'} aria-hidden="true"><Icon size={size} strokeWidth={1.8} /></span>
}
