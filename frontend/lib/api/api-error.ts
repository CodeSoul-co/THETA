/** HTTP 状态保留到调用方，避免把认证或服务故障误当作功能缺失。 */
export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}
