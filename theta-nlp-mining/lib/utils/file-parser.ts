/**
 * 前端文件解析工具
 * 支持 CSV、Excel、JSON 文件的前端解析
 */

export interface ParsedFileData {
  headers: string[]
  rowCount: number
  sampleRows: Record<string, any>[]
  rawData?: any[]
}

/**
 * 解析 CSV 文件
 */
async function parseCSV(file: File): Promise<ParsedFileData> {
  const text = await file.text()
  const lines = text.split('\n').filter(line => line.trim())
  
  if (lines.length === 0) {
    throw new Error('CSV 文件为空')
  }

  // 检测分隔符（逗号或分号）
  const firstLine = lines[0]
  const delimiter = firstLine.includes(',') ? ',' : (firstLine.includes(';') ? ';' : '\t')
  
  // 解析表头
  const headers = lines[0].split(delimiter).map(h => h.trim().replace(/^"|"$/g, ''))
  
  // 解析数据行
  const rows: Record<string, any>[] = []
  for (let i = 1; i < Math.min(lines.length, 101); i++) { // 最多解析 100 行作为示例
    const values = lines[i].split(delimiter).map(v => v.trim().replace(/^"|"$/g, ''))
    const row: Record<string, any> = {}
    headers.forEach((header, idx) => {
      row[header] = values[idx] || ''
    })
    rows.push(row)
  }

  return {
    headers,
    rowCount: lines.length - 1,
    sampleRows: rows.slice(0, 10), // 只返回前 10 行作为示例
    rawData: rows,
  }
}

/**
 * 解析 JSON 文件
 */
async function parseJSON(file: File): Promise<ParsedFileData> {
  const text = await file.text()
  const data = JSON.parse(text)
  
  if (Array.isArray(data)) {
    if (data.length === 0) {
      throw new Error('JSON 数组为空')
    }
    
    const headers = Object.keys(data[0])
    return {
      headers,
      rowCount: data.length,
      sampleRows: data.slice(0, 10),
      rawData: data,
    }
  } else if (typeof data === 'object') {
    const headers = Object.keys(data)
    return {
      headers,
      rowCount: 1,
      sampleRows: [data],
      rawData: [data],
    }
  }
  
  throw new Error('不支持的 JSON 格式')
}

/**
 * 解析 Excel 文件（简单实现，仅支持 CSV 格式的 Excel）
 * 注意：完整的 Excel 解析需要 xlsx 库
 */
async function parseExcel(file: File): Promise<ParsedFileData> {
  // 对于 Excel 文件，我们尝试作为 CSV 解析
  // 实际项目中应该使用 xlsx 库
  return parseCSV(file)
}

/**
 * 解析文件
 */
export async function parseFile(file: File): Promise<ParsedFileData> {
  const fileName = file.name.toLowerCase()
  
  if (fileName.endsWith('.csv')) {
    return parseCSV(file)
  } else if (fileName.endsWith('.json')) {
    return parseJSON(file)
  } else if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
    return parseExcel(file)
  } else {
    throw new Error('不支持的文件格式')
  }
}
